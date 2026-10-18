"""The shell tools: bash copies pi's; run_program is Anthropic's programmatic tool calling."""
import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable

from agent.files import MAX_BYTES, MAX_LINES
from agent.result import Result, ToolError
from agent.sandbox import Sandbox

MAX_TIMEOUT = 600
MAX_CALLS = 100  # tool calls one program may make
LOGS = "/workspace/.agent"  # long output lands here, where read can page through it

BASH = {"type": "function", "function": {
    "name": "bash",
    "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is "
                   "truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is "
                   "saved to a file you can read. Optionally provide a timeout in seconds (default 120, max 600). "
                   "The network reaches only the Python package index.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (default 120, max 600)"},
    }, "required": ["command"]}}}

RUN_PROGRAM = {"type": "function", "function": {
    "name": "run_program",
    "description": "Run a Python program that calls tools, so their results stay out of your context: only what the "
                   "program prints comes back. Inside it, these tools are async functions that take one dict of "
                   "arguments and return the result's text: {tools}. Use top-level await, and asyncio.gather to run "
                   "calls in parallel. A failed call raises ToolError. At most 100 calls per program. Output is "
                   "truncated like bash's. Optionally provide a timeout in seconds (default 120, max 600), inner calls "
                   "included.",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "description": "The Python program"},
        "timeout": {"type": "number", "description": "Timeout in seconds (default 120, max 600)"},
    }, "required": ["code"]}}}

# Prints a log's line and byte counts, then the log whole (and deletes it) or its last lines
# within pi's limits. Only that tail leaves the sandbox.
_TAIL = f"""tail_log() {{
  lines=$(awk 'END {{print NR}}' "$1"); bytes=$(wc -c <"$1")
  echo "$lines $bytes"
  if [ "$lines" -le {MAX_LINES} ] && [ "$bytes" -le {MAX_BYTES} ]; then cat "$1"; rm -f "$1"
  else tail -n {MAX_LINES} "$1" | tail -c {MAX_BYTES}; fi
}}
"""

# $1 the command, $2 the timeout, $3 the log; prints the exit code and the milliseconds it took
_BASH = _TAIL + """mkdir -p "$(dirname "$3")"
start=$(date +%s%N)
timeout -k 5 "$2" bash -c "$1" >"$3" 2>&1 </dev/null
echo "$? $(( ($(date +%s%N) - start) / 1000000 ))"
tail_log "$3"
"""


async def bash(sandbox: Sandbox, command: str, timeout: float = 120) -> Result:
    _check(timeout)
    log = f"{LOGS}/bash-{uuid.uuid4().hex[:8]}.log"
    _, stdout, _ = await sandbox.exec("bash", "-c", _BASH, "bash", command, f"{timeout:g}", log,
                                      timeout=timeout + 30)
    status, rest = stdout.split(b"\n", 1)
    code, milliseconds = map(int, status.split())
    return _output(rest, log, code, milliseconds / 1000, timeout, "Command")


async def run_program(sandbox: Sandbox, run: Callable[[str, dict], Awaitable[Result]], tools: list[str],
                      code: str, timeout: float = 120) -> Result:
    """`run` runs one call from the program. It must check the name against `tools` itself: the
    program shares a process with the stubs that send its calls, so it can send any line."""
    _check(timeout)
    name = uuid.uuid4().hex[:8]
    program, log = f"/tmp/program-{name}.py", f"{LOGS}/program-{name}.log"
    await sandbox.write(program, code.encode())
    start = time.monotonic()
    process = await sandbox.start("python", "-u", "-c", _RUNNER, program, log, json.dumps(tools), timeout=timeout)
    files, sources, calls, lock = [], [], 0, asyncio.Lock()

    async def answer(request: dict) -> None:
        nonlocal calls
        calls += 1
        if calls > MAX_CALLS:
            result = Result(f"run_program allows at most {MAX_CALLS} tool calls.", error=True)
        else:
            result = await run(request["name"], request["arguments"])
        files.extend(result.files)
        sources.extend(result.sources)
        reply = {"id": request["id"], "error" if result.error else "result": result.text()}
        async with lock:
            with contextlib.suppress(Exception):  # the program may have ended meanwhile
                process.stdin.write((json.dumps(reply) + "\n").encode())
                await process.stdin.drain.aio()

    async with asyncio.TaskGroup() as group:
        pending, buffer = set(), b""
        async for chunk in process.stdout:
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                pending.add(group.create_task(answer(json.loads(line))))
        exit_code = await process.wait.aio()
        for task in pending:  # the program is gone; nobody is waiting for these
            task.cancel()
    _, stdout, _ = await sandbox.exec("bash", "-c", _TAIL + 'tail_log "$1"', "bash", log, timeout=30)
    result = _output(stdout, log, exit_code, time.monotonic() - start, timeout, "Program")
    result.files, result.sources = files, sources
    return result


def _check(timeout: float) -> None:
    if not 0 < timeout <= MAX_TIMEOUT:
        raise ToolError(f"timeout must be more than 0 and at most {MAX_TIMEOUT} seconds.")


def _output(tailed: bytes, log: str, code: int, elapsed: float, timeout: float, what: str) -> Result:
    """pi's bash result: the tail of the output, a notice naming the full log when cut, and how it
    ended."""
    counts, shown = tailed.split(b"\n", 1)
    total, size = map(int, counts.split())
    cut = total > MAX_LINES or size > MAX_BYTES  # tail_log kept the log and sent only its tail
    byte_cut = cut and len(shown) == MAX_BYTES
    lines = shown.decode(errors="replace").split("\n")
    if shown.endswith(b"\n"):
        lines.pop()
    if byte_cut and len(lines) > 1:  # tail -c landed inside a line: drop what's left of it
        lines.pop(0)
    out = "\n".join(lines) or "(no output)"
    if cut:
        name = log.removeprefix("/workspace/")
        if byte_cut and len(lines) == 1:
            out += f"\n\n[Showing the last 50.0KB of line {total}. Full output: {name}]"
        else:
            cap = " (50.0KB limit)" if byte_cut else ""
            out += f"\n\n[Showing lines {total - len(lines) + 1}-{total} of {total}{cap}. Full output: {name}]"
    if code in (124, 137) and elapsed >= timeout:
        return Result(out + f"\n\n{what} timed out after {timeout:g} seconds", error=True)
    if code:
        return Result(out + f"\n\n{what} exited with code {code}", error=True)
    return Result(out)


# Runs inside the sandbox. The program's prints go to the log; the runner's original stdout carries
# its calls to the host, one JSON line each, and replies come back on stdin.
_RUNNER = r"""
import ast, asyncio, itertools, json, os, sys, traceback

program, log, names = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
calls = os.fdopen(os.dup(1), "w")
os.makedirs(os.path.dirname(log), exist_ok=True)
output = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
os.dup2(output, 1)
os.dup2(output, 2)


class ToolError(Exception):
    "A tool call that failed; the message says why."


waiting, ids = {}, itertools.count()


async def replies():
    reader = asyncio.StreamReader()
    await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    while line := await reader.readline():
        reply = json.loads(line)
        future = waiting.pop(reply["id"], None)
        if future is None:  # a reply to a call the program forged
            continue
        if "error" in reply:
            future.set_exception(ToolError(reply["error"]))
        else:
            future.set_result(reply["result"])


def tool(name):
    async def call(arguments=None, /, **more):
        future = asyncio.get_running_loop().create_future()
        id = next(ids)
        waiting[id] = future
        calls.write(json.dumps({"id": id, "name": name, "arguments": {**(arguments or {}), **more}}) + "\n")
        calls.flush()
        return await future
    call.__name__ = name
    return call


async def main():
    listener = asyncio.create_task(replies())
    scope = {"__name__": "__main__", "asyncio": asyncio, "ToolError": ToolError, **{name: tool(name) for name in names}}
    with open(program) as f:
        code = compile(f.read(), program, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    try:
        if (coroutine := eval(code, scope)) is not None:
            await coroutine
    finally:
        listener.cancel()


try:
    asyncio.run(main())
except Exception as error:  # show the program's frames, not the runner's
    tb = error.__traceback__
    while tb and tb.tb_frame.f_code.co_filename != program:
        tb = tb.tb_next
    traceback.print_exception(type(error), error, tb)
    sys.exit(1)
"""
