"""run_program's side inside the sandbox. The program's prints go to a log; the runner's original
stdout carries its tool calls to the host, one JSON line each, and the replies come back on stdin.
The program shares this process, so it can write any line: the host checks every call."""
import ast
import asyncio
import itertools
import json
import os
import sys
import traceback


class ToolError(Exception):
    "A tool call that failed; the message says why."


class Bridge:
    def __init__(self, calls, replies_in) -> None:
        self.calls, self.replies_in = calls, replies_in
        self.waiting: dict[int, asyncio.Future] = {}
        self.ids = itertools.count()
        self.broken: ToolError | None = None

    def tool(self, name: str):
        async def call(arguments=None, /, **more):
            if self.broken:
                raise self.broken
            future = asyncio.get_running_loop().create_future()
            id = next(self.ids)
            self.waiting[id] = future
            self.calls.write(json.dumps({"id": id, "name": name, "arguments": {**(arguments or {}), **more}}) + "\n")
            self.calls.flush()
            return await future
        call.__name__ = name
        return call

    async def replies(self) -> None:
        """Hands each reply to its call. If the replies stop or break, every waiting call fails
        rather than hang until the timeout."""
        reader = asyncio.StreamReader(limit=2**30)  # a reply is one line, as long as the tool's whole result
        await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), self.replies_in)
        try:
            while line := await reader.readline():
                reply = json.loads(line)
                future = self.waiting.pop(reply["id"], None)
                if future is None or future.done():  # a call the program forged, or gave up on
                    continue
                if "error" in reply:
                    future.set_exception(ToolError(reply["error"]))
                else:
                    future.set_result(reply["result"])
            self.broken = ToolError("The host stopped answering tool calls.")
        except Exception as error:
            self.broken = ToolError(f"A tool's reply couldn't be read: {error}")
        for future in self.waiting.values():
            if not future.done():
                future.set_exception(self.broken)


async def main(program: str, names: list[str], bridge: Bridge) -> None:
    listener = asyncio.create_task(bridge.replies())
    scope = {"__name__": "__main__", "asyncio": asyncio, "ToolError": ToolError, **{name: bridge.tool(name) for name in names}}
    with open(program) as f:
        code = compile(f.read(), program, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    try:
        if (coroutine := eval(code, scope)) is not None:
            await coroutine
    finally:
        listener.cancel()


if __name__ == "__main__":
    program, log, names = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    # the bridge keeps the original stdin and stdout; the program and anything it starts get /dev/null
    # and the log, so a subprocess can't swallow a reply
    calls, replies_in = os.fdopen(os.dup(1), "w"), os.fdopen(os.dup(0), "rb")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    output = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(output, 1)
    os.dup2(output, 2)
    try:
        asyncio.run(main(program, names, Bridge(calls, replies_in)))
    except Exception as error:  # show the program's frames, not the runner's
        tb = error.__traceback__
        while tb and tb.tb_frame.f_code.co_filename != program:
            tb = tb.tb_next
        traceback.print_exception(type(error), error, tb)
        sys.exit(1)
