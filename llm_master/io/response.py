"""The Response a call returns, read from each API's output. See IO.md."""
import json
from dataclasses import dataclass, field
from typing import Any, Literal, assert_never

from pydantic import TypeAdapter

from llm_master.config.response_format import Format
from llm_master.io.messages import Call, Message, Origin, new_id
from llm_master.providers import API, ProviderName

Finish = Literal["stop", "length", "calls", "refusal"]
_CHAT_FINISH: dict[str, Finish] = {"stop": "stop", "length": "length", "tool_calls": "calls",
                                   "function_call": "calls", "content_filter": "refusal"}
_ANTHROPIC_FINISH: dict[str, Finish] = {"end_turn": "stop", "stop_sequence": "stop", "pause_turn": "stop",
                                        "compaction": "stop", "max_tokens": "length",
                                        "model_context_window_exceeded": "length", "tool_use": "calls",
                                        "refusal": "refusal"}


@dataclass
class Usage:
    """input counts every prompt token, cache reads and writes included; output counts reasoning."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0


@dataclass
class Attempt:
    """One candidate tried, with one typed error per failed try."""

    provider: ProviderName
    model: str
    latency: float = 0.0
    errors: list[Exception] = field(default_factory=list)


@dataclass
class Response:
    """message goes into history; everything else describes this one call."""

    message: Message
    parsed: Any
    finish: Finish
    usage: Usage
    attempts: list[Attempt]
    latency: float
    cost: float | None = None  # SPEC.md step 4


def read(api: API, raw: Any, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage]:
    """One API's output as an assistant message. ids holds wire ids a stream already gave our ids."""
    match api:
        case "completion":
            return _completion(raw, origin, ids)
        case "messages":
            return _anthropic(raw, origin, ids)
        case "responses":
            return _responses(raw, origin, ids)
        case _:
            assert_never(api)


def parse(fmt: Format | None, message: Message, finish: Finish) -> Any:
    """The response_format object, from the same text a stream or a plain call produced."""
    if fmt is None or fmt.type == "text" or finish != "stop" or not isinstance(message.content, str):
        return None
    if isinstance(fmt.schema_, type):
        return TypeAdapter(fmt.schema_).validate_json(message.content)
    return json.loads(message.content)


def _completion(raw: Any, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage]:
    choice = raw.choices[0]
    native = choice.message.model_dump(mode="json", exclude_none=True, exclude={"parsed"})
    calls = [_call(ids, tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
             for tc in native.get("tool_calls", [])]
    text = native.get("content") or native.get("refusal") or ""
    u = raw.usage
    sent, made = (u.prompt_tokens_details, u.completion_tokens_details) if u else (None, None)
    usage = Usage(input=u.prompt_tokens, output=u.completion_tokens,
                  reasoning=made and made.reasoning_tokens or 0, cache_read=sent and sent.cached_tokens or 0,
                  cache_write=sent and sent.cache_write_tokens or 0) if u else Usage()
    return _message(text, calls, native.get("reasoning"), origin, native), _CHAT_FINISH[choice.finish_reason], usage


def _anthropic(raw: Any, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage]:
    native = [b.model_dump(mode="json", exclude_none=True, exclude={"parsed_output"}) for b in raw.content]
    text = "".join(b["text"] for b in native if b["type"] == "text")
    thinking = "\n\n".join(b["thinking"] for b in native if b["type"] == "thinking")
    calls = [_call(ids, b["id"], b["name"], b["input"]) for b in native if b["type"] == "tool_use"]
    u = raw.usage
    cache_read, cache_write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
    usage = Usage(input=u.input_tokens + cache_read + cache_write, output=u.output_tokens,
                  reasoning=u.output_tokens_details.thinking_tokens if u.output_tokens_details else 0,
                  cache_read=cache_read, cache_write=cache_write)
    return _message(text, calls, thinking, origin, native), _ANTHROPIC_FINISH[raw.stop_reason], usage


def _responses(raw: Any, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage]:
    native = [i.model_dump(mode="json", exclude_none=True, exclude={"content": {"__all__": {"parsed"}}})
              for i in raw.output]
    parts = [p for i in native if i["type"] == "message" for p in i["content"]]
    text = "".join(p.get("text") or p.get("refusal") or "" for p in parts)
    summary = "\n\n".join(s["text"] for i in native if i["type"] == "reasoning" for s in i.get("summary", []))
    calls = [_call(ids, i["call_id"], i["name"], json.loads(i["arguments"] or "{}"))
             for i in native if i["type"] == "function_call"]
    if raw.status == "incomplete":
        finish: Finish = "refusal" if raw.incomplete_details.reason == "content_filter" else "length"
    else:
        finish = "calls" if calls else "refusal" if any(p["type"] == "refusal" for p in parts) else "stop"
    u = raw.usage
    usage = Usage(input=u.input_tokens, output=u.output_tokens, reasoning=u.output_tokens_details.reasoning_tokens,
                  cache_read=u.input_tokens_details.cached_tokens,
                  # any-llm's ResponseResource drops it; a stream's openai Response keeps it
                  cache_write=getattr(u.input_tokens_details, "cache_write_tokens", 0))
    return _message(text, calls, summary, origin, native), finish, usage


def _call(ids: dict[str, str], wire_id: str, name: str, arguments: dict) -> Call:
    return Call(id=ids.get(wire_id) or new_id(), name=name, arguments=arguments, wire_id=wire_id)


def _message(text: str, calls: list[Call], reasoning: str | None, origin: Origin, native: Any) -> Message:
    return Message(role="assistant", content=text, calls=calls or None, reasoning=reasoning or None,
                   origin=origin, native=native)
