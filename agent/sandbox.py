"""One Modal sandbox per conversation. /workspace sits on the conversation's own volume, so it
outlives each sandbox: one that idled out is replaced on the next call and finds its files."""
import asyncio
import math
import posixpath
import time

import modal
from modal.exception import AlreadyExistsError, NotFoundError

APP = "llm-master-agent"
WORKSPACE = "/workspace"
IMAGE = (modal.Image.debian_slim(python_version="3.13")
         .apt_install("libreoffice-nogui", "poppler-utils", "fonts-liberation", "fonts-dejavu", "procps")
         .uv_pip_install("pillow", "pillow-heif", "openpyxl"))
IDLE_TIMEOUT = 15 * 60  # seconds without a command before Modal stops it
LIFETIME = 24 * 60 * 60  # Modal's ceiling
MAX_OUTPUT = 64 * 1024 * 1024  # bytes exec takes back; read's largest result, 10 pages as base64, is about 50 MB


class Sandbox:
    """Started on first use. Parallel tool calls share one start."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self._box: modal.Sandbox | None = None
        self._lock = asyncio.Lock()

    async def running(self) -> modal.Sandbox:
        box = self._box
        if box and await box.poll.aio() is None:
            return box
        async with self._lock:
            if self._box and self._box is not box:  # replaced while we waited
                return self._box
            self._box = await _find(self.conversation_id) or await _create(self.conversation_id)
            return self._box

    async def exec(self, *args: str, timeout: float) -> tuple[int, bytes, bytes]:
        """Runs args in /workspace with stdin closed; returns the exit code, stdout and stderr. Past
        `timeout` seconds the command and everything it started are killed, and TimeoutError is
        raised. Past MAX_OUTPUT bytes of output the sandbox is stopped, and OverflowError is raised."""
        box = await self.running()
        start = time.monotonic()
        # coreutils timeout kills the command's whole process group; Modal's deadline, the backstop,
        # kills only the process it started, which ends its output
        process = await box.exec.aio("timeout", "-k", "5", f"{timeout:g}", *args,
                                     timeout=math.ceil(timeout) + 30, text=False)
        process.stdin.write_eof()
        await process.stdin.drain.aio()
        room = MAX_OUTPUT

        async def take(stream) -> bytes:
            nonlocal room
            data = bytearray()
            async for chunk in stream:
                if room < 0:  # the other stream ran over
                    break
                room -= len(chunk)
                if room < 0:  # the model is root in there, so only Modal can be sure to stop it
                    await box.terminate.aio(wait=True)
                    break
                data += chunk
            return bytes(data)

        stdout, stderr = await asyncio.gather(take(process.stdout), take(process.stderr))
        if room < 0:
            raise OverflowError(f"{args[0]} printed over {MAX_OUTPUT} bytes, so its sandbox was stopped")
        code = await process.wait.aio()
        if code and time.monotonic() - start >= timeout:  # 124, 137 or Modal's -1, depending on who killed it
            raise TimeoutError(f"{args[0]} ran past {timeout:g} seconds")
        return code, stdout, stderr

    async def read(self, path: str) -> bytes:
        return await (await self.running()).filesystem.read_bytes.aio(resolve(path))

    async def write(self, path: str, data: bytes) -> None:
        await (await self.running()).filesystem.write_bytes.aio(data, resolve(path))


def resolve(path: str) -> str:
    """The absolute sandbox path for a model's path, relative ones under /workspace."""
    return posixpath.normpath(posixpath.join(WORKSPACE, path))


async def delete(conversation_id: str) -> None:
    """Stops the conversation's sandbox and deletes its files, for good. Stop the conversation's
    tool calls first: a call that lands afterwards starts over on a new, empty volume."""
    box = await _find(conversation_id)
    if box:
        await box.terminate.aio(wait=True)
    await modal.Volume.objects.delete.aio(_volume(conversation_id), allow_missing=True)


def _volume(conversation_id: str) -> str:
    return f"workspace-{conversation_id}"


async def _find(conversation_id: str) -> modal.Sandbox | None:
    try:
        return await modal.Sandbox.from_name.aio(APP, conversation_id)
    except NotFoundError:
        return None


async def _create(conversation_id: str) -> modal.Sandbox:
    app = await modal.App.lookup.aio(APP, create_if_missing=True)
    volume = modal.Volume.from_name(_volume(conversation_id), create_if_missing=True)
    try:
        return await modal.Sandbox.create.aio(app=app, name=conversation_id, image=IMAGE,
                                              volumes={WORKSPACE: volume}, workdir=WORKSPACE,
                                              block_network=True,  # model-written code can't send data out
                                              timeout=LIFETIME, idle_timeout=IDLE_TIMEOUT)
    except AlreadyExistsError:  # another process started it between our lookup and create
        return await modal.Sandbox.from_name.aio(APP, conversation_id)
