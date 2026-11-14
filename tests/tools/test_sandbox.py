"""The per-conversation sandbox on real Modal: one per conversation, files that outlive it, cleanup."""
import asyncio
import uuid

import pytest
from modal.exception import SandboxFilesystemNotFoundError

from agent.sandbox import Sandbox, delete

pytestmark = pytest.mark.live


@pytest.fixture
async def sandbox():
    sandbox = Sandbox(f"test-{uuid.uuid4().hex[:12]}")
    yield sandbox
    await delete(sandbox.conversation_id)


async def test_one_sandbox_per_conversation_across_parallel_calls_and_processes(sandbox):
    other_process = Sandbox(sandbox.conversation_id)
    boxes = await asyncio.gather(sandbox.running(), sandbox.running(), other_process.running())
    assert len({box.object_id for box in boxes}) == 1
    await other_process.write("shared.txt", b"seen")
    assert await sandbox.read("shared.txt") == b"seen"


async def test_workspace_outlives_the_sandbox(sandbox):
    await sandbox.write("notes/plan.txt", b"kept")
    first = await sandbox.running()
    await first.terminate.aio(wait=True)  # what idling out does
    assert await sandbox.exec("cat", "/workspace/notes/plan.txt", timeout=30) == (0, b"kept", b"")
    assert (await sandbox.running()).object_id != first.object_id


async def test_exec_reports_exit_and_output_with_stdin_closed(sandbox):
    assert await sandbox.exec("bash", "-c", "cat; echo out; echo err >&2; exit 3", timeout=30) == (3, b"out\n", b"err\n")


async def test_exec_timeout_kills_everything_the_command_started(sandbox):
    with pytest.raises(TimeoutError):
        await sandbox.exec("bash", "-c", "(sleep 3; touch late) & sleep 10", timeout=1)
    await asyncio.sleep(4)
    code, _, _ = await sandbox.exec("test", "-e", "late", timeout=30)
    assert code == 1


async def test_exec_stops_a_command_that_floods_its_output(sandbox, monkeypatch):
    monkeypatch.setattr("agent.sandbox.MAX_OUTPUT", 1024 * 1024)  # streaming the real 64 MB takes 20 s
    with pytest.raises(OverflowError):
        await sandbox.exec("yes", timeout=600)
    assert await sandbox.exec("echo", "back", timeout=30) == (0, b"back\n", b"")


async def test_image_has_what_the_tools_need(sandbox):
    code, stdout, stderr = await sandbox.exec(
        "bash", "-c", "soffice --version && pdftoppm -v && pandoc --version && ps --version && "
                      "fc-list | grep -q 'Noto Sans CJK' && "
                      "python -c 'import PIL, pillow_heif, openpyxl, charset_normalizer, indra_ocr'",
        timeout=120)
    assert code == 0, (stdout + stderr).decode()


async def test_nothing_leaves_the_sandbox_not_even_dns(sandbox):
    code, _, _ = await sandbox.exec("python", "-c", "import socket; socket.getaddrinfo('example.com', 443)", timeout=60)
    assert code != 0


async def test_delete_removes_the_files(sandbox):
    await sandbox.write("draft.txt", b"gone")
    await delete(sandbox.conversation_id)
    with pytest.raises(SandboxFilesystemNotFoundError):
        await Sandbox(sandbox.conversation_id).read("draft.txt")
