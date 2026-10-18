"""The toolbox: one conversation's tools, listed for the config and run by name."""
import asyncio
import contextlib
import inspect
import logging
from collections.abc import Sequence
from functools import partial

from agent import files, loop, shell, web
from agent.result import Result, ToolError
from agent.sandbox import Sandbox
from agent.web import Web

log = logging.getLogger(__name__)

SERIAL = {"write", "edit", "bash", "run_program"}  # they change the sandbox, so they run one at a time
PROGRAM_TOOLS = ["web_search", "fetch_url", "deliver_file"]  # the host's tools a program may call


class Toolbox:
    def __init__(self, conversation_id: str, sources: Sequence[dict] = (), urls: Sequence[str] = ()) -> None:
        """`sources` are the source parts already sent in this conversation; `urls` the ones the user
        wrote, which fetch_url may read."""
        self.sandbox = Sandbox(conversation_id)
        self.web = Web(self.sandbox, list(sources), urls)
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
        self._tools = {spec["function"]["name"]: (spec, run) for spec, run in builtins}
        self._serial = asyncio.Lock()

    @property
    def specs(self) -> list[dict]:
        """Every tool as an openai function dict, for Config(tools=...)."""
        return [spec for spec, _ in self._tools.values()]

    async def run(self, name: str, arguments: dict) -> Result:
        """Runs one call. Never raises: a failure comes back as an error result the model can read."""
        try:
            return await self._call(name, arguments)
        except ToolError as error:
            return Result(str(error), error=True)
        except Exception as error:  # ours, not the model's: the message may hold what the model mustn't see
            log.exception("tool %s failed", name)
            return Result(f"{name} failed ({type(error).__name__}). The details are in the server log.", error=True)

    async def resume(self, name: str, arguments: dict, reply: dict[str, str] | None) -> Result:
        """Finishes a call that came back waiting, with the user's reply: ask_user's answers, keyed by
        question, or None when the user declined."""
        assert name == "ask_user", f"{name} never waits"
        return Result(loop.answered(reply))

    async def aclose(self) -> None:
        await self.web.aclose()

    async def _call(self, name: str, arguments: dict) -> Result:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        run = self._tools[name][1]
        try:
            inspect.signature(run).bind(**arguments)
        except TypeError as error:
            raise ToolError(f"Invalid arguments for {name}: {error}") from None
        async with self._serial if name in SERIAL else contextlib.nullcontext():
            out = await run(**arguments) if inspect.iscoroutinefunction(run) else await asyncio.to_thread(run, **arguments)
        return out if isinstance(out, Result) else Result(str(out))

    async def _run_from_program(self, name: str, arguments: dict) -> Result:
        """A call a program made. Checked here, not in the sandbox, where the program can write any call."""
        if name not in self._program_tools:
            return Result(f"{name} can't be called from a program.", error=True)
        return await self.run(name, arguments)
