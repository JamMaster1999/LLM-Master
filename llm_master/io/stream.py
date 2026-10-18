"""Streams: each API's events as AI SDK UI message stream chunks. See IO.md Streams."""
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, assert_never

from any_llm.exceptions import ProviderError
from any_llm.types.completion import ChatCompletion

from llm_master.io.messages import new_id
from llm_master.io.response import Response
from llm_master.providers import API


class Stream:
    """Chunks the browser's useChat reads. response holds the Response once the loop ends; it is never yielded."""

    def __init__(self, events: AsyncGenerator[dict | Response, None]):
        self._events = events
        self.response: Response | None = None

    async def __aiter__(self) -> AsyncIterator[dict]:
        async for event in self._events:
            if isinstance(event, Response):
                self.response = event
            else:
                yield event

    async def aclose(self) -> None:
        """Abort: closes the provider's stream."""
        await self._events.aclose()


def chunks(api: API, events: AsyncIterator[Any], ids: dict[str, str]) -> AsyncIterator[Any]:
    """The step's chunks, then the provider's complete output for response.read. ids collects wire id → our id."""
    match api:
        case "completion":
            return _completion(events, ids)
        case "messages":
            return _anthropic(events, ids)
        case "responses":
            return _responses(events, ids)
        case _:
            assert_never(api)


class _Blocks:
    """The step's open text or reasoning block: a delta of another kind ends it and starts its own."""

    def __init__(self):
        self.kind, self.id = "", ""

    def feed(self, kind: str, delta: str) -> list[dict]:
        out = []
        if kind != self.kind:
            out = self.close()
            self.kind, self.id = kind, new_id()
            out.append({"type": f"{kind}-start", "id": self.id})
        return [*out, {"type": f"{kind}-delta", "id": self.id, "delta": delta}]

    def close(self) -> list[dict]:
        out = [{"type": f"{self.kind}-end", "id": self.id}] if self.kind else []
        self.kind = ""
        return out


def _start(call_id: str, name: str) -> dict:
    return {"type": "tool-input-start", "toolCallId": call_id, "toolName": name}


def _delta(call_id: str, text: str) -> dict:
    return {"type": "tool-input-delta", "toolCallId": call_id, "inputTextDelta": text}


def _available(call_id: str, name: str, arguments: dict) -> dict:
    return {"type": "tool-input-available", "toolCallId": call_id, "toolName": name, "input": arguments}


def _cut() -> ProviderError:
    return ProviderError("the stream ended before the response finished")


async def _completion(events: AsyncIterator[Any], ids: dict[str, str]) -> AsyncIterator[Any]:
    blocks, calls = _Blocks(), {}
    message: dict[str, Any] = {"role": "assistant"}
    usage = finish = None
    async for chunk in events:
        usage = chunk.usage or usage  # gemini sends a running total on every chunk: the last one counts
        for choice in chunk.choices:
            finish = choice.finish_reason or finish
            delta = choice.delta.model_dump(mode="json", exclude_none=True, exclude={"role"})
            if delta.get("reasoning"):
                for c in blocks.feed("reasoning", delta["reasoning"]):
                    yield c
            if delta.get("content"):
                for c in blocks.feed("text", delta["content"]):
                    yield c
            for tc in delta.pop("tool_calls", []):
                fn = tc.pop("function", {})
                if tc["index"] not in calls:
                    for c in blocks.close():
                        yield c
                    calls[tc["index"]] = {"function": {"name": fn.get("name", ""), "arguments": ""}}
                    ids[tc["id"]] = new_id()
                    yield _start(ids[tc["id"]], fn.get("name", ""))
                call = calls[tc["index"]]
                call.update(tc)
                if fn.get("arguments"):
                    call["function"]["arguments"] += fn["arguments"]
                    yield _delta(ids[call["id"]], fn["arguments"])
            for key, value in delta.items():  # text concatenates; gemini's last text signature wins
                message[key] = message.get(key, "") + value if isinstance(value, str) else value
    if finish is None:
        raise _cut()
    for c in blocks.close():
        yield c
    for call in calls.values():
        yield _available(ids[call["id"]], call["function"]["name"], json.loads(call["function"]["arguments"] or "{}"))
    if calls:
        message["tool_calls"] = list(calls.values())
    yield ChatCompletion.model_validate({"id": "", "created": 0, "model": "", "object": "chat.completion", "usage": usage,
                                         "choices": [{"index": 0, "finish_reason": finish, "message": message}]})


async def _anthropic(events: AsyncIterator[Any], ids: dict[str, str]) -> AsyncIterator[Any]:
    blocks, tools, final = _Blocks(), {}, None
    async for e in events:
        if e.type == "content_block_start" and e.content_block.type == "tool_use":
            for c in blocks.close():
                yield c
            tools[e.index] = ids[e.content_block.id] = new_id()
            yield _start(tools[e.index], e.content_block.name)
        elif e.type == "content_block_delta" and e.delta.type == "thinking_delta":
            for c in blocks.feed("reasoning", e.delta.thinking):
                yield c
        elif e.type == "content_block_delta" and e.delta.type == "text_delta":
            for c in blocks.feed("text", e.delta.text):
                yield c
        elif e.type == "content_block_delta" and e.delta.type == "input_json_delta" and e.delta.partial_json:
            yield _delta(tools[e.index], e.delta.partial_json)
        elif e.type == "content_block_stop" and e.index in tools:
            yield _available(tools[e.index], e.content_block.name, e.content_block.input)
        elif e.type == "content_block_stop":
            for c in blocks.close():
                yield c
        elif e.type == "message_stop":
            final = e.message  # the SDK's accumulated message: same reading as a plain call
    if final is None:
        raise _cut()
    yield final


async def _responses(events: AsyncIterator[Any], ids: dict[str, str]) -> AsyncIterator[Any]:
    blocks, tools, final = _Blocks(), {}, None
    async for e in events:
        if e.type == "response.output_item.added" and e.item.type == "function_call":
            for c in blocks.close():
                yield c
            tools[e.item.id] = ids[e.item.call_id] = new_id()
            yield _start(tools[e.item.id], e.item.name)
        elif e.type == "response.function_call_arguments.delta":
            yield _delta(tools[e.item_id], e.delta)
        elif e.type == "response.output_item.done" and e.item.type == "function_call":
            yield _available(tools[e.item.id], e.item.name, json.loads(e.item.arguments or "{}"))
        elif e.type == "response.reasoning_summary_text.delta":
            for c in blocks.feed("reasoning", e.delta):
                yield c
        elif e.type in ("response.output_text.delta", "response.refusal.delta"):
            for c in blocks.feed("text", e.delta):
                yield c
        elif e.type in ("response.reasoning_summary_part.done", "response.content_part.done"):
            for c in blocks.close():
                yield c
        elif e.type in ("response.completed", "response.incomplete"):
            final = e.response
        elif e.type in ("response.failed", "error"):
            error = e.response.error if e.type == "response.failed" else e
            raise ProviderError(error.message if error else "the response failed", provider_name="openai")
    if final is None:
        raise _cut()
    yield final
