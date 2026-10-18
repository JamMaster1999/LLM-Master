"""Streams: each API's events as AI SDK UI message stream chunks. See IO.md Streams."""
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, assert_never

from any_llm.exceptions import ProviderError
from any_llm.types.completion import ChatCompletion

from llm_master.io.messages import new_id
from llm_master.io.response import Response, arguments
from llm_master.providers import API


class Stream:
    """Chunks the browser's useChat reads. response holds the Response once the loop ends; it is never yielded.

    Abort by leaving the loop and awaiting aclose(), or by cancelling the task that iterates: either way the
    provider's stream closes. Python can't close a generator another task is inside, so aclose() belongs
    to the iterating side.
    """

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
        await self._events.aclose()


def adapter(api: API, ids: dict[str, str]) -> "_Completion | _Messages | _Responses":
    """Turns one API's events into chunks; ids collects each call's wire id → our id."""
    match api:
        case "completion":
            return _Completion(ids)
        case "messages":
            return _Messages(ids)
        case "responses":
            return _Responses(ids)
        case _:
            assert_never(api)


def _cut() -> ProviderError:
    return ProviderError("the stream ended before the response finished")


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


class _Tools:
    """Each call's input chunks under our lm_ id, created when the call starts streaming."""

    def __init__(self, ids: dict[str, str]):
        self.ids = ids

    def start(self, wire_id: str, name: str) -> dict:
        self.ids[wire_id] = new_id()
        return {"type": "tool-input-start", "toolCallId": self.ids[wire_id], "toolName": name}

    def delta(self, wire_id: str, text: str) -> dict:
        return {"type": "tool-input-delta", "toolCallId": self.ids[wire_id], "inputTextDelta": text}

    def available(self, wire_id: str, name: str, args: dict) -> dict:
        return {"type": "tool-input-available", "toolCallId": self.ids[wire_id], "toolName": name, "input": args}


class _Completion:
    """Chat completion deltas, joined into the message any-llm would have returned whole."""

    def __init__(self, ids: dict[str, str]):
        self.blocks, self.tools = _Blocks(), _Tools(ids)
        self.message: dict[str, Any] = {"role": "assistant"}
        self.calls: dict[int, dict] = {}
        self.usage = self.finish_reason = None

    def feed(self, chunk: Any) -> list[dict]:
        self.usage = chunk.usage or self.usage  # gemini sends a running total on every chunk: the last one counts
        out: list[dict] = []
        for choice in chunk.choices[:1]:  # one choice is asked for; a plain call reads the first
            self.finish_reason = choice.finish_reason or self.finish_reason
            delta = choice.delta.model_dump(mode="json", exclude_none=True, exclude={"role"})
            if delta.get("reasoning"):
                out += self.blocks.feed("reasoning", delta["reasoning"])
            if delta.get("content") or delta.get("refusal"):  # a refusal's text is the answer, as in a plain call
                out += self.blocks.feed("text", delta.get("content") or delta["refusal"])
            for part in delta.pop("tool_calls", []):
                out += self._call(part)
            for key, value in delta.items():  # text concatenates; gemini's last text signature wins
                self.message[key] = self.message.get(key, "") + value if isinstance(value, str) else value
        return out

    def _call(self, part: dict) -> list[dict]:
        fn, out = part.pop("function", {}), []
        if part["index"] not in self.calls:  # the first fragment carries the id
            out = [*self.blocks.close(), self.tools.start(part["id"], fn.get("name", ""))]
            self.calls[part["index"]] = {"function": {"name": "", "arguments": ""}}
        call = self.calls[part["index"]]
        call.update(part)  # id, type, index and gemini's signature in extra_content
        call["function"]["name"] += fn.get("name") or ""
        call["function"]["arguments"] += fn.get("arguments") or ""
        if fn.get("arguments"):
            out.append(self.tools.delta(call["id"], fn["arguments"]))
        return out

    def finish(self) -> tuple[list[dict], ChatCompletion]:
        if self.finish_reason is None or self.usage is None:  # fireworks sends usage after the finish
            raise _cut()
        out = self.blocks.close()
        for call in self.calls.values():
            out.append(self.tools.available(call["id"], call["function"]["name"],
                                            arguments(call["function"]["arguments"])))
        if self.calls:
            self.message["tool_calls"] = list(self.calls.values())
        return out, ChatCompletion.model_validate({
            "id": "", "created": 0, "model": "", "object": "chat.completion", "usage": self.usage,
            "choices": [{"index": 0, "finish_reason": self.finish_reason, "message": self.message}]})


class _Messages:
    """Anthropic's events; its SDK hands over the whole message at message_stop."""

    def __init__(self, ids: dict[str, str]):
        self.blocks, self.tools = _Blocks(), _Tools(ids)
        self.calls: dict[int, str] = {}  # block index → wire id
        self.final = None

    def feed(self, e: Any) -> list[dict]:
        if e.type == "content_block_start" and e.content_block.type == "tool_use":
            self.calls[e.index] = e.content_block.id
            return [*self.blocks.close(), self.tools.start(e.content_block.id, e.content_block.name)]
        if e.type == "content_block_delta" and e.delta.type == "thinking_delta" and e.delta.thinking:
            return self.blocks.feed("reasoning", e.delta.thinking)
        if e.type == "content_block_delta" and e.delta.type == "text_delta":
            return self.blocks.feed("text", e.delta.text)
        if e.type == "content_block_delta" and e.delta.type == "input_json_delta" and e.delta.partial_json:
            return [self.tools.delta(self.calls[e.index], e.delta.partial_json)]
        if e.type == "content_block_stop" and e.index in self.calls:
            return [self.tools.available(self.calls[e.index], e.content_block.name, e.content_block.input)]
        if e.type == "content_block_stop":
            return self.blocks.close()
        if e.type == "message_stop":
            self.final = e.message
        return []

    def finish(self) -> tuple[list[dict], Any]:
        if self.final is None:
            raise _cut()
        return [], self.final


class _Responses:
    """OpenAI's events; response.completed carries the whole response."""

    def __init__(self, ids: dict[str, str]):
        self.blocks, self.tools = _Blocks(), _Tools(ids)
        self.calls: dict[str, str] = {}  # item id → wire id (call_id)
        self.final = None

    def feed(self, e: Any) -> list[dict]:
        if e.type == "response.output_item.added" and e.item.type == "function_call":
            self.calls[e.item.id] = e.item.call_id
            return [*self.blocks.close(), self.tools.start(e.item.call_id, e.item.name)]
        if e.type == "response.function_call_arguments.delta":
            return [self.tools.delta(self.calls[e.item_id], e.delta)]
        if e.type == "response.output_item.done" and e.item.type == "function_call":
            return [self.tools.available(e.item.call_id, e.item.name, arguments(e.item.arguments))]
        if e.type == "response.reasoning_summary_text.delta":
            return self.blocks.feed("reasoning", e.delta)
        if e.type in ("response.output_text.delta", "response.refusal.delta"):
            return self.blocks.feed("text", e.delta)
        if e.type in ("response.reasoning_summary_part.done", "response.content_part.done"):
            return self.blocks.close()
        if e.type in ("response.completed", "response.incomplete"):
            self.final = e.response
        if e.type in ("response.failed", "error"):
            error = e.response.error if e.type == "response.failed" else e
            raise ProviderError(error.message if error else "the response failed", provider_name="openai")
        return []

    def finish(self) -> tuple[list[dict], Any]:
        if self.final is None:
            raise _cut()
        return [], self.final
