"""The backend on Modal at its entry points, the HTTP endpoint and the CLI, on an app of each test
worker's own."""
import asyncio
import os
import secrets
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import modal
import pytest
from modal.exception import NotFoundError

from agent.sandbox import Sandbox
from agent.turn import STOPPED, Conversation
from server import MODELS

ROOT = Path(__file__).parent.parent.parent
# server/app.py reads these at import: each test worker deploys its own app, public while the tests
# run, with a short prompt and the fixture skills
os.environ.update(AGENT_APP=f"llm-master-test-{secrets.token_hex(4)}", AGENT_PUBLIC="1",
                  AGENT_PROMPT=str(ROOT / "tests" / "fixtures" / "prompt.md"),
                  AGENT_SKILLS=str(ROOT / "tests" / "tools" / "skills"))

from server.app import app, load, locks, web  # noqa: E402
from server.cli import chunks  # noqa: E402

pytestmark = pytest.mark.live
plumbing = pytest.mark.xdist_group("backend")  # the cheap ones share a worker, and so one app
MODEL = "claude-sonnet-5-5"


@pytest.fixture(scope="session")
def url() -> Iterator[str]:
    with app.run():
        address = web.get_web_url()
        assert address
        yield address


@pytest.fixture
def conversation(url) -> Iterator[str]:
    id = f"test-{uuid.uuid4().hex[:12]}"
    yield id
    httpx.delete(f"{url}/conversations/{id}", timeout=120).raise_for_status()


def said(url: str, conversation: str, text: str) -> list[dict]:
    body = {"id": conversation, "model": MODEL, "message": {"role": "user", "parts": [{"type": "text", "text": text}]}}
    with httpx.Client(timeout=httpx.Timeout(30, read=None)) as client:
        return list(chunks(client, url, body))


@plumbing
def test_a_message_streams_its_answer(url, conversation):
    streamed = said(url, conversation, "Reply with the word ok.")
    assert (streamed[0]["type"], streamed[-1]["type"]) == ("start", "finish")
    assert "ok" in "".join(c["delta"] for c in streamed if c["type"] == "text-delta").lower()


@pytest.mark.agentic
@pytest.mark.parametrize("model", list(MODELS))  # each provider's key and stream through the deployed app
def test_the_cli_uploads_asks_and_saves_what_the_agent_sends(url, conversation, tmp_path, model):
    (tmp_path / "notes.txt").write_text("The secret word is tangerine.\n")
    script = [f"/attach {tmp_path / 'notes.txt'}",
              "Use ask_user to ask me one question: which file name to use, with the options word.txt and "
              "other.txt. Then write the secret word from the file I attached into a file with that name, and send "
              "it with deliver_file.",
              "1",  # the first option
              "/quit"]
    run = subprocess.run([sys.executable, "-m", "server.cli", "--url", url, "--conversation", conversation,
                          "--model", model, "--files", str(tmp_path / "files")],
                         input="\n".join(script) + "\n", capture_output=True, text=True, timeout=600, cwd=ROOT)
    saved = tmp_path / "files" / "word.txt"
    assert f"Saved {saved}" in run.stdout and "tangerine" in saved.read_text(), run.stdout + run.stderr


@pytest.mark.agentic
def test_a_stopped_turn_answers_its_open_call_and_the_next_request_waits_for_it(url, conversation):
    with httpx.Client(timeout=httpx.Timeout(30, read=None)) as client:
        body = {"id": conversation, "model": MODEL, "message": {"role": "user", "parts": [
            {"type": "text", "text": "Run `sleep 120` with bash, then say done."}]}}
        for chunk in chunks(client, url, body):
            if chunk["type"] == "tool-input-available":
                break  # the user stops the turn while the command runs
    answer = "".join(c["delta"] for c in said(url, conversation, "Reply with the word ok.") if c["type"] == "text-delta")
    messages = asyncio.run(load(conversation)).messages
    assert [m.content for m in messages if m.role == "tool"] == [STOPPED] and "ok" in answer.lower()


@plumbing
def test_delete_removes_the_files_and_the_history(url, conversation):
    said(url, conversation, "Reply with the word ok.")  # a history
    asyncio.run(Sandbox(conversation).write("keep.txt", b"hi"))  # and files, as a tool would leave them
    httpx.delete(f"{url}/conversations/{conversation}", timeout=120).raise_for_status()
    assert asyncio.run(load(conversation)) == Conversation()
    with pytest.raises(NotFoundError):
        modal.Volume.from_name(f"workspace-{conversation}").hydrate()


@plumbing
def test_a_lock_whose_holder_died_is_taken_over(url, conversation):
    locks.put(conversation, ("a dead container", time.time() - 1))
    answer = "".join(c["delta"] for c in said(url, conversation, "Reply with the word ok.") if c["type"] == "text-delta")
    assert "ok" in answer.lower()
