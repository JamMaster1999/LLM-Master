"""The shell tools: bash copies pi's; run_program is Anthropic's programmatic tool calling."""
import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from importlib import resources
from typing import Annotated

from modal.exception import ConflictError
from pydantic import Field

from agent.reader import MAX_BYTES, MAX_FILE, MAX_LINES, format_size
from agent.result import Result
from agent.sandbox import SCRATCH, Sandbox

MAX_CALLS = 100  # tool calls one program may make
MAX_CALL_BYTES = 1024 * 1024  # one call's JSON line; a program sending more is stopped
MAX_ARGUMENT = 60_000  # bytes; Modal takes 65,536 characters across an exec's arguments, Linux 131,072 bytes in each
STDIN_CHUNK = 16 * 1024 * 1024  # Modal buffers at most this much stdin between drains
Timeout = Annotated[float, Field(gt=0, le=600)]
_RUNNER = resources.files("agent").joinpath("runner.py").read_text()

BASH = {"type": "function", "function": {
    "name": "bash",
    "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is "
                   "truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is "
                   "saved to a file you can read. Optionally provide a timeout in seconds (default 120, max 600). "
                   "The sandbox has no network.",
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


async def bash(sandbox: Sandbox, command: str, timeout: Timeout = 120) -> Result:
    name = uuid.uuid4().hex[:8]
    log = f"{SCRATCH}/bash-{name}.log"
    if len(command.encode()) > MAX_ARGUMENT:  # Modal and Linux cap exec arguments, and a heredoc can pass that
        await sandbox.write(f"/tmp/bash-{name}.sh", command.encode())
        command = f"source /tmp/bash-{name}.sh"
    _, stdout, _ = await sandbox.exec("bash", "-c", _BASH, "bash", command, f"{timeout:g}", log,
                                      timeout=timeout + 30)
    status, rest = stdout.split(b"\n", 1)
    code, milliseconds = map(int, status.split())
    return _output(rest, log, code, milliseconds / 1000, timeout, "Command")


async def run_program(sandbox: Sandbox, run: Callable[[str, dict], Awaitable[Result]], tools: list[str],
                      code: str, timeout: Timeout = 120) -> Result:
    """`run` runs one call from the program. Every check is here, on the host: the program shares
    a process with the stubs that send its calls, so it can send any line."""
    name = uuid.uuid4().hex[:8]
    program, log = f"/tmp/program-{name}.py", f"{SCRATCH}/program-{name}.log"
    await sandbox.write(program, code.encode())
    start = time.monotonic()
    # -I keeps /workspace off the import path, so a random.py there can't break the runner
    process = await sandbox.start("python", "-I", "-u", "-c", _RUNNER, program, log, json.dumps(tools), timeout=timeout)
    files, sources, lock, unsent = [], [], asyncio.Lock(), bytearray()

    async def answer(request: dict, number: int) -> None:
        if request["name"] not in tools:
            result = Result(f"{request['name']} can't be called from a program.", error=True)
        elif number > MAX_CALLS:
            result = Result(f"run_program allows at most {MAX_CALLS} tool calls.", error=True)
        else:
            result = await run(request["name"], request["arguments"])
            if sum(len(file.data) for file in files + result.files) > MAX_FILE:  # all on the host until the program ends
                result = Result(f"run_program sends the user at most {format_size(MAX_FILE)} of files.", error=True)
        files.extend(result.files)
        sources.extend(result.sources)
        reply = {"id": request["id"], "error" if result.error else "result": result.text()}
        unsent.extend((json.dumps(reply) + "\n").encode())
        async with lock:  # the replies that land during a write go out together in the next, one round trip
            data = bytes(unsent)
            unsent.clear()
            with contextlib.suppress(ConflictError):  # the program ended meanwhile
                for start in range(0, len(data), STDIN_CHUNK):
                    process.stdin.write(data[start:start + STDIN_CHUNK])
                    await process.stdin.drain.aio()

    async with asyncio.TaskGroup() as group:
        pending, buffer, received = set(), b"", 0
        async for chunk in process.stdout:
            *lines, buffer = (buffer + chunk).split(b"\n")
            received += len(lines)
            # it's flooding the bridge, not calling tools
            if received > 2 * MAX_CALLS or max(map(len, [buffer, *lines])) > MAX_CALL_BYTES:
                await sandbox.exec("pkill", "-f", program, timeout=30)
                break
            for number, line in enumerate(lines, received - len(lines) + 1):
                with contextlib.suppress(ValueError, KeyError, TypeError):  # a forged line that isn't a call
                    request = json.loads(line)
                    task = group.create_task(answer({"id": request["id"], "name": request["name"],
                                                     "arguments": dict(request["arguments"])}, number))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
        exit_code = await process.wait.aio()
        for task in pending:  # the program is gone; nobody is waiting for these
            task.cancel()
    _, stdout, _ = await sandbox.exec("bash", "-c", _TAIL + 'tail_log "$1"', "bash", log, timeout=30)
    result = _output(stdout, log, exit_code, time.monotonic() - start, timeout, "Program")
    result.files, result.sources = files, sources
    return result


def _output(tailed: bytes, log: str, code: int, elapsed: float, timeout: float, what: str) -> Result:
    """pi's bash result: the tail of the output, a notice naming the full log when cut, and how it
    ended."""
    counts, shown = tailed.split(b"\n", 1)
    total, size = map(int, counts.split())
    shown = shown[-MAX_BYTES:]  # tail_log cut it already, unless the model, root in there, swapped the tools it runs
    cut = total > MAX_LINES or size > MAX_BYTES  # tail_log kept the log and sent only its tail
    byte_cut = cut and len(shown) == MAX_BYTES
    lines = shown.decode(errors="replace").split("\n")
    if shown.endswith(b"\n"):
        lines.pop()
    if byte_cut and len(lines) > 1:  # tail -c landed inside a line: drop what's left of it
        lines.pop(0)
    lines = lines[-MAX_LINES:]
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
