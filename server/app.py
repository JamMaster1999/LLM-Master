"""The agent's backend on Modal: a web endpoint that runs a conversation's turns and streams them in the
AI SDK UI message stream protocol. See TOOLS.md, The backend.

    modal deploy --env dev server/app.py

The deploy takes the system prompt from AGENT_PROMPT and the skills from AGENT_SKILLS
(prompts/system.md and skills/ by default) and names the app AGENT_APP. AGENT_PUBLIC=1 drops Modal's
proxy auth, for an app that lives only as long as a test.
"""
import asyncio
import base64
import binascii
import contextlib
import io
import json
import logging
import mimetypes
import os
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import fastapi
import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import AfterValidator, BaseModel, Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict  # pydantic takes typing's only from Python 3.12

from agent import UserFile, sandbox
from agent.turn import Conversation, turn
from llm_master.facade import LLM
from llm_master.types import Model
from server import APP, MODELS

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PROMPT, SKILLS = Path("/root/prompt.md"), Path("/root/skills")  # where the image keeps what the deploy chose
NAME = r"^[A-Za-z0-9._-]{1,54}$"  # a conversation id names its sandbox and volume (TOOLS.md, The sandbox)
TIMEOUT = 60 * 60  # seconds one request may run
LEASE = 60  # seconds a conversation's lock lasts unless its request renews it
WAIT = 30  # seconds a request waits for the one before it to wind up
PING = 5  # seconds of quiet before a ping: Modal tells the app a client left only when a write fails
MAX_FILES = 20  # per message (TOOLS.md, Files)

image = (modal.Image.debian_slim(python_version="3.13")
         .apt_install("git")  # uv fetches any-llm from its fork
         .uv_sync(str(ROOT), extras=["agent", "server"], extra_options="--no-dev")
         .add_local_python_source("agent", "llm_master", "server")
         .add_local_file(os.environ.get("AGENT_PROMPT", ROOT / "prompts/system.md"), str(PROMPT)))
if (skills := Path(os.environ.get("AGENT_SKILLS", ROOT / "skills"))).is_dir():
    image = image.add_local_dir(skills, str(SKILLS))

app = modal.App(os.environ.get("AGENT_APP", APP))
llm = LLM()  # one per container, so requests reuse its connections
history = modal.Volume.from_name("llm-master-history", create_if_missing=True, version=2)
locks = modal.Dict.from_name("llm-master-locks", create_if_missing=True)
api = FastAPI()


# nonpreemptible, at 3x the container's price: a preempted request reruns with no client to stream to
@app.function(image=image, secrets=[modal.Secret.from_name("llm-master-keys")], timeout=TIMEOUT, nonpreemptible=True)
@modal.asgi_app(requires_proxy_auth=os.environ.get("AGENT_PUBLIC") != "1")
def web() -> FastAPI:
    return api


@app.function(image=sandbox.image(skills) if skills.is_dir() else sandbox.IMAGE)
def sandbox_image() -> None:
    """Never called: it makes the deploy build the sandbox's image, so a package list that doesn't
    install fails the deploy, and the first sandbox after it starts without a build."""


class Part(BaseModel):
    """The fields of an AI SDK UI message part the backend reads."""

    type: str
    text: str = ""
    url: str = ""
    mediaType: str = ""
    filename: str | None = None
    toolCallId: str = ""
    state: str = ""
    output: Any = None


class UIMessage(BaseModel):
    role: Literal["user", "assistant"]
    parts: list[Part]


def _offered(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"the backend offers {', '.join(MODELS)}")
    return model


class Chat(BaseModel):
    """The body the AI SDK sends when it sends only the last message, and the model to answer it."""

    id: str = Field(pattern=NAME)
    message: UIMessage
    model: Annotated[str, AfterValidator(_offered)]


class Said(TypedDict, total=False):
    """turn()'s arguments for one message."""

    text: str
    uploads: list[tuple[str, bytes]]
    answers: dict[str, dict[str, str] | None]


Answers: TypeAdapter[dict[str, str] | None] = TypeAdapter(dict[str, str] | None)


class _Closing(StreamingResponse):
    """A StreamingResponse that closes its events however it ends. Starlette leaves them open when the
    client leaves mid-write, and then the turn would wind up and save only when the garbage collector
    got to it, after the request, when Modal no longer runs the event loop."""

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            assert isinstance(self.body_iterator, AsyncGenerator)
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()


@api.post("/chat")
async def chat(request: Chat) -> StreamingResponse:
    """One turn: a user message, or the assistant message that carries ask_user's answers."""
    model = MODELS[request.model]
    model = model.model_copy(update={"config": model.config.model_copy(update={"system": PROMPT.read_text()})})
    return _Closing(_events(request.id, model, _input(request.message)), media_type="text/event-stream",
                    headers={"x-vercel-ai-ui-message-stream": "v1"})


@api.delete("/conversations/{id}")
async def forget(id: Annotated[str, fastapi.Path(pattern=NAME)]) -> None:
    """Deletes a conversation for good: its sandbox, its files and its history. Uflo calls it when
    the user deletes the conversation."""
    try:
        async with _locked(id):
            await sandbox.delete(id)
            with contextlib.suppress(FileNotFoundError):
                await history.remove_file.aio(f"{id}.json")
    except Busy as error:
        raise HTTPException(409, str(error)) from None


def _input(message: UIMessage) -> Said:
    """turn()'s arguments for the message: a user message's text and uploads, or ask_user's answers
    by call id. An answer is the answers by question, or null when the user declined."""
    if message.role == "assistant":
        try:
            return {"answers": {part.toolCallId: Answers.validate_python(part.output) for part in message.parts
                                if part.type == "tool-ask_user" and part.state == "output-available"}}
        except ValidationError:
            raise HTTPException(422, "ask_user's output must map each question to its answer.") from None
    files = [part for part in message.parts if part.type == "file"]
    if len(files) > MAX_FILES:
        raise HTTPException(422, f"A message takes at most {MAX_FILES} files.")
    return {"text": "\n\n".join(part.text for part in message.parts if part.type == "text"),
            "uploads": [(part.filename or f"upload{mimetypes.guess_extension(part.mediaType) or ''}", _decoded(part.url))
                        for part in files]}


def _decoded(url: str) -> bytes:
    """A file part's bytes, from the base64 data URL the AI SDK makes of an upload."""
    head, _, data = url.partition(",")
    if not (head.startswith("data:") and head.endswith(";base64")):
        raise HTTPException(422, "A file part's url must be a base64 data URL.")
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error:
        raise HTTPException(422, "A file part's url holds broken base64.") from None


async def _events(id: str, model: Model, said: Said) -> AsyncGenerator[str, None]:
    """The turn as server-sent events, ending in [DONE]."""
    try:
        async with _conversation(id) as conversation:
            turned = turn(llm, model, id, conversation, skills=SKILLS if SKILLS.is_dir() else None, **said)
            async with aclosing(_pinged(turned)) as events:
                async for event in events:
                    yield event
    except Busy as error:
        yield _event({"type": "error", "errorText": str(error)})
    yield "data: [DONE]\n\n"


async def _pinged(chunks: AsyncGenerator[dict | UserFile, None]) -> AsyncGenerator[str, None]:
    """The chunks as events, with a ping after PING quiet seconds. The turn runs in a task of its own,
    so the pings go on while it waits on a tool. A client that leaves cancels it, and the events end
    once it has wound up. FastAPI's EventSourceResponse pings only every 15 seconds, and cancels
    in anyio's way inside the turn's cleanup."""
    queue: asyncio.Queue[dict | UserFile | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async with aclosing(chunks):
                async for chunk in chunks:
                    queue.put_nowait(chunk)
        except Exception as error:
            log.exception("turn failed")
            queue.put_nowait({"type": "error", "errorText": f"{type(error).__name__}: {error}"})
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), PING)
            except TimeoutError:
                yield ": ping\n\n"  # a comment, which the AI SDK's parser skips
                continue
            if chunk is None:
                break
            yield _event(chunk)
    finally:
        task.cancel()  # a no-op once the turn is over
        with anyio.CancelScope(shield=True):  # Starlette cancels in anyio's way, again at every await
            await asyncio.wait([task])


def _event(chunk: dict | UserFile) -> str:
    """One server-sent event. A file goes as a data URL until the files origin exists, its name in
    providerMetadata: the AI SDK's file chunk has no field for it."""
    if isinstance(chunk, UserFile):
        chunk = {"type": "file", "mediaType": chunk.mime,
                 "url": f"data:{chunk.mime};base64,{base64.b64encode(chunk.data).decode()}",
                 "providerMetadata": {"agent": {"filename": chunk.name}}}
    return f"data: {json.dumps(chunk)}\n\n"  # ASCII, so a lone UTF-16 half from a model can't break it


class Busy(Exception):
    """Another request is running in the conversation."""


@asynccontextmanager
async def _locked(id: str) -> AsyncIterator[None]:
    """One request per conversation at a time (TOOLS.md). The lock is a Dict entry naming its holder
    and when it lapses, which the holder pushes back while it runs, so a container that dies frees the
    conversation soon. A lapsed lock passes to the first request to claim it, under a key named for it,
    so two can't both take it."""
    me, deadline = uuid.uuid4().hex, time.monotonic() + WAIT
    while not await locks.put.aio(id, (me, time.time() + LEASE), skip_if_exists=True):
        held = await locks.get.aio(id)
        if held and held[1] < time.time() and await locks.put.aio(f"{id}/{held[0]}", me, skip_if_exists=True):
            await locks.put.aio(id, (me, time.time() + LEASE))
            break
        if time.monotonic() > deadline:
            raise Busy("Another request is still running in this conversation.")
        await asyncio.sleep(0.5)

    async def mine() -> bool:
        held = await locks.get.aio(id)
        return bool(held) and held[0] == me

    async def renew() -> None:
        while True:
            await asyncio.sleep(LEASE / 3)
            try:
                if not await mine():
                    log.error("lost the lock on %s", id)
                    return
                await locks.put.aio(id, (me, time.time() + LEASE))
            except Exception:  # Modal blinked; the lease has two more renewals before it lapses
                log.exception("couldn't renew the lock on %s", id)

    renewing = asyncio.create_task(renew())
    try:
        yield
    finally:
        renewing.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.wait([renewing])
            if await mine():
                await locks.pop.aio(id, None)


@asynccontextmanager
async def _conversation(id: str) -> AsyncIterator[Conversation]:
    """The conversation's state, locked to this request, and saved after it, however it ends."""
    async with _locked(id):
        conversation = await load(id)
        try:
            yield conversation
        finally:
            with anyio.CancelScope(shield=True):
                await _save(id, conversation)


async def load(id: str) -> Conversation:
    """A conversation's state, from the history volume, which no sandbox mounts."""
    try:
        data = b"".join([chunk async for chunk in history.read_file.aio(f"{id}.json")])
    except FileNotFoundError:
        return Conversation()
    return Conversation.model_validate_json(data)


async def _save(id: str, conversation: Conversation) -> None:
    data = conversation.model_dump_json(exclude_none=True).encode()
    async with history.batch_upload.aio(force=True) as batch:
        batch.put_file(io.BytesIO(data), f"{id}.json")
