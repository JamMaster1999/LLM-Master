"""The toolbox: one conversation's tools, listed for the config and run by name."""
import asyncio
import contextlib
import inspect
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from functools import partial

from any_llm.tools import callable_to_tool
from pydantic import ConfigDict, ValidationError
from pydantic.experimental.arguments_schema import generate_arguments_schema
from pydantic_core import ArgsKwargs, SchemaValidator

from agent import files, loop, reader, shell, web
from agent.result import Result, ToolError
from agent.sandbox import SCRATCH, Sandbox
from agent.web import Web

log = logging.getLogger(__name__)

SERIAL = {"write", "edit", "bash", "run_program"}  # they change the sandbox, so they run one at a time
PROGRAM_TOOLS = ["web_search", "fetch_url", "deliver_file"]  # the host's tools a program may call; custom ones join
CUT_THEIR_OWN = {"read", "bash", "run_program"}  # every other tool's long text is cut here


class Toolbox:
    def __init__(self, conversation_id: str, *, tools: Sequence[Callable] = (), sources: Sequence[dict] = (),
                 urls: Sequence[str] = ()) -> None:
        """`tools` are the caller's own functions; `sources` the source parts already sent in this
        conversation; `urls` the ones the user wrote, which fetch_url may read."""
        self.sandbox = Sandbox(conversation_id)
        self.web = Web(list(sources), urls)
        custom = [(callable_to_tool(tool), tool) for tool in tools]
        program_tools = PROGRAM_TOOLS + [spec["function"]["name"] for spec, _ in custom]
        run_program = dict(shell.RUN_PROGRAM, function={**shell.RUN_PROGRAM["function"], "description":
                           shell.RUN_PROGRAM["function"]["description"].format(tools=", ".join(program_tools))})
        builtins = [
            (files.READ, partial(files.read, self.sandbox)),
            (files.WRITE, partial(files.write, self.sandbox)),
            (files.EDIT, partial(files.edit, self.sandbox)),
            (shell.BASH, partial(shell.bash, self.sandbox)),
            (run_program, partial(shell.run_program, self.sandbox, self._run_from_program, program_tools)),
            (web.SEARCH, self.web.search),
            (web.FETCH, self.web.fetch),
            (files.DELIVER, partial(files.deliver_file, self.sandbox)),
            (loop.TODO, loop.todo),
            (loop.ASK_USER, loop.ask_user),
        ]
        names = [spec["function"]["name"] for spec, _ in builtins + custom]
        if repeated := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"Tool names must be unique; repeated: {', '.join(repeated)}")
        self._tools = {name: (spec, *_checked(run)) for name, (spec, run) in zip(names, builtins + custom)}
        self._serial = asyncio.Lock()

    @property
    def specs(self) -> list[dict]:
        """Every tool as an openai function dict, for Config(tools=...)."""
        return [spec for spec, _, _ in self._tools.values()]

    async def run(self, name: str, arguments: dict) -> Result:
        """Runs one call. Never raises: a failure comes back as an error result the model can read."""
        return await self._run(name, arguments, cut=True)

    async def resume(self, name: str, arguments: dict, reply: dict[str, str] | None) -> Result:
        """Finishes a call that came back waiting, with the user's reply: ask_user's answers, keyed by
        question, or None when the user declined."""
        assert name == "ask_user", f"{name} never waits"
        return Result(loop.answered(reply))

    async def aclose(self) -> None:
        await self.web.aclose()

    async def _run(self, name: str, arguments: dict, cut: bool) -> Result:
        try:
            result = await self._call(name, arguments)
            if cut and isinstance(result.content, str) and name not in CUT_THEIR_OWN:
                result.content = await self._cut(name, result.content)
            return result
        except ToolError as error:
            return Result(str(error), error=True)
        except Exception as error:  # ours, not the model's: the message may hold what the model mustn't see
            log.exception("tool %s failed", name)
            return Result(f"{name} failed ({type(error).__name__}). The details are in the server log.", error=True)

    async def _call(self, name: str, arguments: dict) -> Result:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        _, check, run = self._tools[name]
        try:  # the check alone: a ValidationError the tool itself raises is its bug, not bad arguments
            args, kwargs = check.validate_python(ArgsKwargs((), arguments))
        except ValidationError as error:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
            raise ToolError(f"Invalid arguments for {name}: {problems}") from None
        async with self._serial if name in SERIAL else contextlib.nullcontext():
            out = await run(*args, **kwargs)
        if isinstance(out, Result):
            return out
        return Result(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))

    async def _cut(self, name: str, text: str) -> str:
        """pi's cut for a long result: the start, and the whole text saved in the sandbox for read."""
        if text.count("\n") < reader.MAX_LINES and len(text.encode()) <= reader.MAX_BYTES:
            return text
        path = f"{SCRATCH}/{name}-{uuid.uuid4().hex[:8]}.txt"
        await self.sandbox.write(path, text.encode())
        shown = path.removeprefix("/workspace/")
        return f"{reader.page(text, shown, None, None)}\n[The whole result is at {shown}.]"

    async def _run_from_program(self, name: str, arguments: dict) -> Result:
        """A call a program made, which run_program has checked. Not cut: the program sees
        everything, and only what it prints reaches the model."""
        return await self._run(name, arguments, cut=False)


def _checked(run: Callable) -> tuple[SchemaValidator, Callable[..., Awaitable]]:
    """validate_call's check of the tool's arguments against its type hints, without the call, and
    the tool as an async function. A wrong type or a missing argument is the model's to fix, and
    models send "30" for 30."""
    schema = generate_arguments_schema(run, "arguments", config=ConfigDict(arbitrary_types_allowed=True))
    return SchemaValidator(schema), run if inspect.iscoroutinefunction(run) else partial(asyncio.to_thread, run)
