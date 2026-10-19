"""The toolbox: one conversation's tools, listed for the config and run by name."""
import asyncio
import contextlib
import inspect
import logging
import uuid
from collections.abc import Callable, Sequence
from functools import partial

from pydantic import ConfigDict, ValidationError, validate_call

from agent import files, loop, reader, shell, web
from agent.result import Result, ToolError
from agent.sandbox import SCRATCH, Sandbox
from agent.web import Web

log = logging.getLogger(__name__)

SERIAL = {"write", "edit", "bash", "run_program"}  # they change the sandbox, so they run one at a time
PROGRAM_TOOLS = ["web_search", "fetch_url", "deliver_file"]  # the host's tools a program may call
CUT_THEIR_OWN = {"read", "bash", "run_program"}  # every other tool's long text is cut here


class Toolbox:
    def __init__(self, conversation_id: str, sources: Sequence[dict] = (), urls: Sequence[str] = ()) -> None:
        """`sources` are the source parts already sent in this conversation; `urls` the ones the user
        wrote, which fetch_url may read."""
        self.sandbox = Sandbox(conversation_id)
        self.web = Web(list(sources), urls)
        self._program_tools = PROGRAM_TOOLS
        run_program = dict(shell.RUN_PROGRAM, function={**shell.RUN_PROGRAM["function"], "description":
                           shell.RUN_PROGRAM["function"]["description"].format(tools=", ".join(self._program_tools))})
        builtins = [
            (files.READ, partial(files.read, self.sandbox)),
            (files.WRITE, partial(files.write, self.sandbox)),
            (files.EDIT, partial(files.edit, self.sandbox)),
            (shell.BASH, partial(shell.bash, self.sandbox)),
            (run_program, partial(shell.run_program, self.sandbox, self._run_from_program, self._program_tools)),
            (web.SEARCH, self.web.search),
            (web.FETCH, self.web.fetch),
            (files.DELIVER, partial(files.deliver_file, self.sandbox)),
            (loop.TODO, loop.todo),
            (loop.ASK_USER, loop.ask_user),
        ]
        self.tools = {spec["function"]["name"]: (spec, run) for spec, run in builtins}  # each one's schema and function
        self._checked = {name: _checked(run) for name, (_, run) in self.tools.items()}
        self._serial = asyncio.Lock()

    @property
    def specs(self) -> list[dict]:
        """Every tool as an openai function dict, for Config(tools=...). Not strict: OpenAI makes a
        tool strict by default, which turns its optional parameters into required ones."""
        return [{**spec, "function": {**spec["function"], "strict": False}} for spec, _ in self.tools.values()]

    async def run(self, name: str, arguments: dict) -> Result:
        """Runs one call. Never raises: a failure comes back as an error result the model can read."""
        return await self._run(name, arguments, cut=True)

    def resume(self, reply: dict[str, str] | None) -> Result:
        """Finishes the ask_user call that came back waiting, with the user's answers keyed by
        question, or None when the user declined."""
        return Result(loop.answered(reply))

    async def aclose(self) -> None:
        await self.web.aclose()

    async def _run(self, name: str, arguments: dict, cut: bool) -> Result:
        try:
            try:
                result = await self._call(name, arguments)
            except ToolError as error:
                result = Result(str(error), error=True)
            if cut and name not in CUT_THEIR_OWN:
                result.content = await self._cut(name, result.content)
            return result
        except Exception as error:  # ours, not the model's: the message may hold what the model mustn't see
            log.exception("tool %s failed", name)
            return Result(f"{name} failed ({type(error).__name__}). The details are in the server log.", error=True)

    async def _call(self, name: str, arguments: dict) -> Result:
        if name not in self.tools:
            raise ToolError(f"Unknown tool: {name}")
        async with self._serial if name in SERIAL else contextlib.nullcontext():
            try:
                out = await self._checked[name](**arguments)
            except ValidationError as error:
                problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
                raise ToolError(f"Invalid arguments for {name}: {problems}") from None
        return out if isinstance(out, Result) else Result(str(out))

    async def _cut(self, name: str, content: str | list[dict]) -> str | list[dict]:
        """Long text cut at pi's limits, keeping the start even mid-line, and saved whole in the sandbox."""
        if isinstance(content, list):
            return [{**part, "text": await self._cut(name, part["text"])} if part["type"] == "text" else part
                    for part in content]
        start = "\n".join(content.split("\n", reader.MAX_LINES)[:reader.MAX_LINES])
        start = start.encode()[:reader.MAX_BYTES].decode(errors="ignore")
        if start == content:
            return content
        path = f"{SCRATCH}/{name}-{uuid.uuid4().hex[:8]}.txt"
        await self.sandbox.write(path, content.encode())
        shown, of = path.removeprefix("/workspace/"), reader.format_size(len(content.encode()))
        return (f"{start}\n\n[Showing the first {reader.format_size(len(start.encode()))} of {of}. "
                f"The whole result is at {shown}; use read or bash for the rest.]")

    async def _run_from_program(self, name: str, arguments: dict) -> Result:
        """A call a program made, which run_program has checked. Not cut: the program sees
        everything, and only what it prints reaches the model."""
        return await self._run(name, arguments, cut=False)


def _checked(run: Callable) -> Callable:
    """The tool as one async function that checks its arguments against its type hints first: a
    wrong type or a missing argument is the model's to fix, and models send "30" for 30."""
    checked = validate_call(config=ConfigDict(arbitrary_types_allowed=True))(run)
    if inspect.iscoroutinefunction(run):
        return checked
    return lambda **arguments: asyncio.to_thread(checked, **arguments)
