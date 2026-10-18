"""Streams: each API's events as AI SDK UI message stream chunks. See IO.md Streams."""
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, assert_never

from any_llm.exceptions import ProviderError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage
from any_llm.types.messages import MessageResponse, MessageStreamEvent
from openai.types.responses import Response as OpenAIResponse, ResponseStreamEvent

from llm_master.io.messages import new_id
from llm_master.io.response import Response, arguments
from llm_master.pipeline import Result
from llm_master.providers import API


class Stream:
    """Chunks the browser's useChat reads. response holds the Response once the loop ends; it is never yielded.

    However the loop ends, by break, an error or a cancel inside its body, the provider's stream closes.
    aclose() closes it at once; Python can't close a generator another task is inside, so aclose()
    belongs to the iterating side.
    """

    def __init__(self, events: AsyncGenerator[dict | Result[Response], None]):
        self._events = events
        self.response: Response | None = None

    async def __aiter__(self) -> AsyncIterator[dict]:
        # a loop that dies early drops this generator, and its finalizer closes the events
        async with aclosing(self._events) as events:
            async for event in events:
                if isinstance(event, Result):
                    self.response = event.value
                else:
                    yield event

    async def aclose(self) -> None:
        await self._events.aclose()


def adapter(api: API) -> "_Completion | _Messages | _Responses":
    """Turns one API's events into chunks; its tools.ids map each call's wire id to our id."""
    match api:
        case "completion":
            return _Completion()
        case "messages":
            return _Messages()
        case "responses":
            return _Responses()
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

    def __init__(self):
        self.ids: dict[str, str] = {}

    def start(self, wire_id: str, name: str) -> dict:
        self.ids[wire_id] = new_id()
        return {"type": "tool-input-start", "toolCallId": self.ids[wire_id], "toolName": name}

    def delta(self, wire_id: str, text: str) -> dict:
        return {"type": "tool-input-delta", "toolCallId": self.ids[wire_id], "inputTextDelta": text}

    def available(self, wire_id: str, name: str, text: str) -> dict:
        """The call's whole arguments, parsed strictly: a call cut off mid-arguments raises CutCall here."""
        return {"type": "tool-input-available", "toolCallId": self.ids[wire_id], "toolName": name,
                "input": arguments(text)}


class _Completion:
    """Chat completion deltas, joined into the message any-llm would have returned whole."""

    def __init__(self):
        self.blocks, self.tools = _Blocks(), _Tools()
        self.message: dict[str, Any] = {"role": "assistant"}
        self.calls: dict[int, dict] = {}
        self.usage: CompletionUsage | None = None
        self.finish_reason: str | None = None

    def feed(self, chunk: ChatCompletionChunk) -> list[dict]:
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
            out.append(self.tools.available(call["id"], call["function"]["name"], call["function"]["arguments"]))
        if self.calls:
            self.message["tool_calls"] = list(self.calls.values())
        return out, ChatCompletion.model_validate({
            "id": "", "created": 0, "model": "", "object": "chat.completion", "usage": self.usage,
            "choices": [{"index": 0, "finish_reason": self.finish_reason, "message": self.message}]})


class _Messages:
    """Anthropic's events; its SDK hands over the whole message at message_stop."""

    def __init__(self):
        self.blocks, self.tools = _Blocks(), _Tools()
        self.calls: dict[int, dict[str, str]] = {}  # block index → the call's wire id, name and JSON so far
        self.final: MessageResponse | None = None

    def feed(self, e: MessageStreamEvent) -> list[dict]:
        if e.type == "content_block_start" and e.content_block.type == "tool_use":
            self.calls[e.index] = {"id": e.content_block.id, "name": e.content_block.name, "json": ""}
            return [*self.blocks.close(), self.tools.start(e.content_block.id, e.content_block.name)]
        if e.type == "content_block_delta" and e.delta.type == "thinking_delta" and e.delta.thinking:
            return self.blocks.feed("reasoning", e.delta.thinking)
        if e.type == "content_block_delta" and e.delta.type == "text_delta":
            return self.blocks.feed("text", e.delta.text)
        if e.type == "content_block_delta" and e.delta.type == "input_json_delta" and e.delta.partial_json:
            self.calls[e.index]["json"] += e.delta.partial_json
            return [self.tools.delta(self.calls[e.index]["id"], e.delta.partial_json)]
        if e.type == "content_block_stop" and e.index in self.calls:
            # not the SDK's block: it parses cut-off JSON leniently. A call with no arguments streams none.
            call = self.calls[e.index]
            return [self.tools.available(call["id"], call["name"], call["json"] or "{}")]
        if e.type == "content_block_stop":
            return self.blocks.close()
        if e.type == "message_stop":
            self.final = e.message
        return []

    def finish(self) -> tuple[list[dict], MessageResponse]:
        if self.final is None:
            raise _cut()
        return [], self.final


class _Responses:
    """OpenAI's events; response.completed carries the whole response."""

    def __init__(self):
        self.blocks, self.tools = _Blocks(), _Tools()
        self.calls: dict[int, str] = {}  # output index → wire id (call_id)
        self.final: OpenAIResponse | None = None

    def feed(self, e: ResponseStreamEvent) -> list[dict]:
        if e.type == "response.output_item.added" and e.item.type == "function_call":
            self.calls[e.output_index] = e.item.call_id
            return [*self.blocks.close(), self.tools.start(e.item.call_id, e.item.name)]
        if e.type == "response.function_call_arguments.delta":
            return [self.tools.delta(self.calls[e.output_index], e.delta)]
        if e.type == "response.output_item.done" and e.item.type == "function_call":
            return [self.tools.available(e.item.call_id, e.item.name, e.item.arguments)]
        if e.type == "response.reasoning_summary_text.delta":
            return self.blocks.feed("reasoning", e.delta)
        if e.type == "response.output_text.delta" or e.type == "response.refusal.delta":
            return self.blocks.feed("text", e.delta)
        if e.type in ("response.reasoning_summary_part.done", "response.content_part.done"):
            return self.blocks.close()
        if e.type == "response.completed" or e.type == "response.incomplete" or e.type == "response.failed":
            self.final = e.response  # read() raises for a failed one, as for a plain call
        if e.type == "error":
            raise ProviderError(e.message, provider_name="openai")
        return []

    def finish(self) -> tuple[list[dict], OpenAIResponse]:
        if self.final is None:
            raise _cut()
        return [], self.final
