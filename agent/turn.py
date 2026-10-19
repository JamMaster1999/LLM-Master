"""The loop: one request's turn in a conversation. The model's calls run in parallel on the toolbox and
their results go back to it, until it answers without a call or asks the user. See TOOLS.md, The loop."""
import asyncio
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from agent.result import Result, UserFile
from agent.toolbox import SERIAL, Toolbox
from llm_master.facade import LLM
from llm_master.io.messages import Call, Message
from llm_master.types import Model

STOPPED = "Stopped by the user."
INTERRUPTED = "[Request interrupted by user]"  # what Claude Code tells the model after a stop mid-answer
URL = re.compile(r"https?://[^\s<>\"'`]+")


class Conversation(BaseModel):
    """What a conversation keeps between requests. Keep it where the sandbox can't write: the model
    takes its history as the truth. A waiting ask_user call is the last assistant message's
    unanswered call."""

    messages: list[Message] = []
    sources: list[dict] = []  # the source parts already sent, so the numbers go on across requests
    urls: list[str] = []  # the URLs the user typed, which fetch_url may read


async def turn(llm: LLM, model: Model, conversation_id: str, conversation: Conversation, *,
               text: str | None = None, uploads: Sequence[tuple[str, bytes]] = (),
               answers: Mapping[str, dict[str, str] | None] = {}, skills: str | Path | None = None,
               ) -> AsyncGenerator[dict | UserFile, None]:
    """Runs one request: AI SDK UI message stream chunks, start and finish included, and the files
    for the user, which the caller hosts. Close it to stop the turn. A provider's failure raises,
    as in llm_master, and the caller sends the error chunk.

    `text` and `uploads`, (name, bytes) pairs, make a new user message. `answers` finish waiting
    ask_user calls, by call id; a waiting call without one is declined. The turn appends to
    `conversation` as it goes. A stopped one answers each open call with "Stopped by the user.", and
    one stopped mid-answer keeps the text so far, then "[Request interrupted by user]", so the
    conversation is always ready for the next request."""
    if text:
        conversation.urls += [url for url in typed_urls(text) if url not in conversation.urls]
    box = Toolbox(conversation_id, sources=conversation.sources, urls=conversation.urls, skills=skills, llm=llm)
    model = model.model_copy(update={"config": model.config.model_copy(update={"tools": box.specs})})
    ended = False
    try:
        yield {"type": "start"}
        for call in _open(conversation.messages):
            conversation.messages.append(_answer(call, box.resume(answers.get(call.id))))
        if text or uploads:
            parts = [{"type": "text", "text": text}] if text else []
            for name, data in uploads:
                parts += await box.attach(name, data)
            conversation.messages.append(Message.model_validate({"role": "user", "content": parts}))
        while True:
            streamed = ""
            try:
                stream = await llm.query(model, conversation.messages, stream=True)
                async for chunk in stream:
                    if chunk["type"] == "text-delta":
                        streamed += chunk["delta"]
                    yield chunk
            except (GeneratorExit, asyncio.CancelledError):  # stopped mid-answer: keep the text the user saw
                conversation.messages += [Message(role="assistant", content=streamed)] if streamed else []
                conversation.messages.append(Message(role="user", content=INTERRUPTED))
                raise
            assert stream.response is not None  # set once the stream ends
            message = stream.response.message
            conversation.messages.append(message)
            asking = False
            for batch in _batches(message.calls or []):
                async with asyncio.TaskGroup() as group:
                    runs = [group.create_task(_run(box, call)) for call in batch]
                    for run in asyncio.as_completed(runs):
                        call, result = await run
                        if result.waiting:  # the browser asks the user; the answers come with the next request
                            asking = True
                            continue
                        conversation.messages.append(_answer(call, result))
                        for chunk in _shown(call, result, conversation.sources):
                            yield chunk
            if asking or not message.calls:
                break
        ended = True
        yield {"type": "finish"}
    finally:
        if not ended:  # stopped, or failed: llm_master raises, as for a normal call
            for call in _open(conversation.messages):
                conversation.messages.append(Message(role="tool", call_id=call.id, content=STOPPED))
        await box.aclose()


def typed_urls(text: str) -> list[str]:
    """The URLs in what the user typed, without the punctuation that follows them. A closing
    parenthesis stays when the URL opened one, as Wikipedia's do."""
    urls = []
    for url in URL.findall(text):
        url = url.rstrip(".,;:!?*")
        if url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        urls.append(url)
    return urls


def _batches(calls: list[Call]) -> list[list[Call]]:
    """The calls in the order they run, as Claude Code runs them: a call that changes the sandbox
    runs alone, after the calls before it, and the calls between such calls run together. A
    review_file sent beside the bash that builds its file then reviews the new file."""
    batches: list[list[Call]] = []
    for call in calls:
        if batches and call.name not in SERIAL and batches[-1][-1].name not in SERIAL:
            batches[-1].append(call)
        else:
            batches.append([call])
    return batches


async def _run(box: Toolbox, call: Call) -> tuple[Call, Result]:
    return call, await box.run(call.name, call.arguments)


def _shown(call: Call, result: Result, sent: list[dict]) -> list[dict | UserFile]:
    """What the user sees of a result: its text as the call's output, its new sources and its files."""
    output = ({"type": "tool-output-error", "toolCallId": call.id, "errorText": result.text()} if result.error
              else {"type": "tool-output-available", "toolCallId": call.id, "output": result.text()})
    new = [part for part in result.sources if part not in sent]
    sent += new
    return [output, *new, *result.files]


def _answer(call: Call, result: Result) -> Message:
    return Message.model_validate({"role": "tool", "call_id": call.id, "content": result.content})


def _open(messages: list[Message]) -> list[Call]:
    """The calls of the last assistant message that have no answer yet."""
    last = next((i for i in reversed(range(len(messages))) if messages[i].role == "assistant"), None)
    if last is None:
        return []
    answered = {m.call_id for m in messages[last + 1:]}
    return [call for call in messages[last].calls or [] if call.id not in answered]
