"""One Modal sandbox per conversation. /workspace sits on the conversation's own volume, so it
outlives each sandbox: one that idled out is replaced on the next call and finds its files."""
import asyncio
import math
import posixpath
import time
from pathlib import Path

import modal
from modal.container_process import ContainerProcess
from modal.exception import AlreadyExistsError, NotFoundError
from modal.types import FileInfo

APP = "llm-master-agent"
WORKSPACE = "/workspace"
SCRATCH = f"{WORKSPACE}/.agent"  # where long output waits for read
SKILLS = "/skills"  # the caller's skill folders, the Claude API's layout
INDRA_MODELS = "/opt/indra-ocr"
IMAGE = (modal.Image.debian_slim(python_version="3.13")  # Debian bookworm
         .run_commands("echo 'deb http://deb.debian.org/debian bookworm-backports main' "
                       "> /etc/apt/sources.list.d/backports.list",
                       "apt-get update && apt-get install -y -t bookworm-backports libreoffice-nogui")  # 25.2, not 7.4
         .apt_install("poppler-utils", "pandoc", "fonts-liberation", "fonts-dejavu", "fonts-crosextra-carlito",
                      "fonts-crosextra-caladea", "fonts-noto-core", "procps")
         .uv_pip_install("pillow", "pillow-heif", "openpyxl", "charset-normalizer", "indra-ocr==0.3.2", "pypdfium2")
         # the weights, since the sandbox can't download them
         .run_commands(f"python -c 'import indra_ocr, shutil; shutil.copytree(indra_ocr.model_dir(), \"{INDRA_MODELS}\")'")
         .env({"INDRA_OCR_MODEL_DIR": INDRA_MODELS}))
CPUS = 4  # Indra reads pages in parallel
IDLE_TIMEOUT = 15 * 60  # seconds without a command before Modal stops it
LIFETIME = 24 * 60 * 60  # Modal's ceiling
MAX_OUTPUT = 64 * 1024 * 1024  # bytes exec takes back; read's largest result, 10 pages as base64, is about 50 MB


class Sandbox:
    """Started on first use. Parallel tool calls share one start. `skills`, a folder of skill
    folders, appears at /skills."""

    def __init__(self, conversation_id: str, skills: str | Path | None = None) -> None:
        self.conversation_id = conversation_id
        self.image = image(Path(skills)) if skills else IMAGE
        self._box: modal.Sandbox | None = None
        self._lock = asyncio.Lock()

    async def running(self) -> modal.Sandbox:
        box = self._box
        if box and await box.poll.aio() is None:
            return box
        async with self._lock:
            if self._box and self._box is not box:  # replaced while we waited
                return self._box
            self._box = await _find(self.conversation_id) or await _create(self.conversation_id, self.image)
            return self._box

    async def start(self, *args: str, timeout: float) -> ContainerProcess:
        """Starts args in /workspace. Past `timeout` seconds the command and everything it started
        are killed, and it exits 124 (137 if it ignored the first signal)."""
        box = await self.running()
        # coreutils timeout kills the command's whole process group; Modal's deadline, the backstop,
        # kills only the process it started, which ends its output
        return await box.exec.aio("timeout", "-k", "5", f"{timeout:g}", *args,
                                  timeout=math.ceil(timeout) + 30, text=False)

    async def exec(self, *args: str, timeout: float) -> tuple[int, bytes, bytes]:
        """Runs args with stdin closed; returns the exit code, stdout and stderr. Raises TimeoutError
        past `timeout` seconds. Past MAX_OUTPUT bytes of output the sandbox is stopped, and
        OverflowError is raised."""
        box = await self.running()  # the clock starts once the sandbox has
        start = time.monotonic()
        process = await self.start(*args, timeout=timeout)
        process.stdin.write_eof()
        await process.stdin.drain.aio()
        room = MAX_OUTPUT

        async def take(stream) -> bytes:
            nonlocal room
            data = bytearray()
            async for chunk in stream:
                room -= len(chunk)
                if room < 0:  # the model is root in there, so only Modal can be sure to stop it
                    await box.terminate.aio(wait=True)
                    raise OverflowError(f"{args[0]} printed over {MAX_OUTPUT} bytes, so its sandbox was stopped")
                data += chunk
            return bytes(data)

        try:  # an overflow cancels the other stream's reader, which Modal would leave waiting for seconds
            async with asyncio.TaskGroup() as group:
                stdout, stderr = group.create_task(take(process.stdout)), group.create_task(take(process.stderr))
        except* Exception as failed:
            raise failed.exceptions[0] from None
        code = await process.wait.aio()
        if code and time.monotonic() - start >= timeout:  # 124, 137 or Modal's -1, depending on who killed it
            raise TimeoutError(f"{args[0]} ran past {timeout:g} seconds")
        return code, stdout.result(), stderr.result()

    async def stat(self, path: str) -> FileInfo:
        return await (await self.running()).filesystem.stat.aio(resolve(path))

    async def read(self, path: str) -> bytes:
        return await (await self.running()).filesystem.read_bytes.aio(resolve(path))

    async def write(self, path: str, data: bytes) -> None:
        await (await self.running()).filesystem.write_bytes.aio(data, resolve(path))


def image(skills: Path) -> modal.Image:
    """IMAGE with the packages the skills list, apt.txt's Debian ones and requirements.txt's pip ones,
    and the skills themselves at /skills."""
    built = IMAGE
    if (apt := skills / "apt.txt").is_file():  # one package per line; # starts a comment
        packages = [line.split("#")[0].strip() for line in apt.read_text().splitlines()]
        built = built.apt_install(*filter(None, packages))
    if (pip := skills / "requirements.txt").is_file():
        built = built.uv_pip_install(requirements=[str(pip)])
    # copy=False, the default: Modal adds the files as the sandbox starts, without a build
    return built.add_local_dir(skills, SKILLS)


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


async def _create(conversation_id: str, image: modal.Image) -> modal.Sandbox:
    app = await modal.App.lookup.aio(APP, create_if_missing=True)
    volume = modal.Volume.from_name(_volume(conversation_id), create_if_missing=True)
    try:
        return await modal.Sandbox.create.aio(app=app, name=conversation_id, image=image,
                                              volumes={WORKSPACE: volume}, workdir=WORKSPACE, cpu=CPUS,
                                              block_network=True,  # model-written code can't send data out
                                              timeout=LIFETIME, idle_timeout=IDLE_TIMEOUT)
    except AlreadyExistsError:  # another process started it between our lookup and create
        return await modal.Sandbox.from_name.aio(APP, conversation_id)
