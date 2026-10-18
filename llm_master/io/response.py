"""The Response a call returns, read from each API's output. See IO.md."""
import json
from dataclasses import dataclass, field
from typing import Any, Literal, assert_never

from anthropic.types.parsed_message import ParsedMessage
from any_llm.exceptions import ProviderError
from any_llm.types.completion import ChatCompletion, CompletionUsage
from any_llm.types.messages import MessageResponse
from openai.types.responses import Response as OpenAIResponse
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
    usage: Usage | None  # None when the provider reported none: unknown, never zero
    attempts: list[Attempt]
    latency: float
    cost: float | None = None  # SPEC.md step 4


def read(api: API, raw: Any, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage | None]:
    """One API's output as an assistant message. ids holds wire ids a stream already gave our ids."""
    match api:
        case "completion":
            return _completion(raw, origin, ids)
        case "messages":
            return _messages(raw, origin, ids)
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


def arguments(text: str) -> dict:
    """A call's JSON arguments. A call cut off mid-arguments (max_tokens) can't run, so it raises typed."""
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError as e:
        raise ProviderError(f"a tool call's arguments aren't complete JSON, so it can't run: {text[:80]!r}") from e


def completion_usage(u: CompletionUsage | None) -> Usage | None:
    """any-llm's chat usage; gemini's media calls report it too."""
    if u is None:
        return None
    sent, made = u.prompt_tokens_details, u.completion_tokens_details
    return Usage(input=u.prompt_tokens, output=u.completion_tokens, reasoning=made and made.reasoning_tokens or 0,
                 cache_read=sent and sent.cached_tokens or 0, cache_write=sent and sent.cache_write_tokens or 0)


def _completion(raw: ChatCompletion, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage | None]:
    choice = raw.choices[0]
    native = choice.message.model_dump(mode="json", exclude_none=True, exclude={"parsed"})
    calls = [_call(ids, tc["id"], tc["function"]["name"], arguments(tc["function"]["arguments"]))
             for tc in native.get("tool_calls", [])]
    text = native.get("content") or native.get("refusal") or ""
    finish = _CHAT_FINISH[choice.finish_reason]
    return _message(text, calls, native.get("reasoning"), origin, native), finish, completion_usage(raw.usage)


def _messages(raw: MessageResponse | ParsedMessage, origin: Origin,
              ids: dict[str, str]) -> tuple[Message, Finish, Usage]:
    native = [b.model_dump(mode="json", exclude_none=True, exclude={"parsed_output"}) for b in raw.content]
    text = "".join(b["text"] for b in native if b["type"] == "text")
    thinking = "\n\n".join(b["thinking"] for b in native if b["type"] == "thinking")
    calls = [_call(ids, b["id"], b["name"], b["input"]) for b in native if b["type"] == "tool_use"]
    u = raw.usage
    cache_read, cache_write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
    usage = Usage(input=u.input_tokens + cache_read + cache_write, output=u.output_tokens,
                  reasoning=u.output_tokens_details.thinking_tokens if u.output_tokens_details else 0,
                  cache_read=cache_read, cache_write=cache_write)
    return _message(text, calls, thinking, origin, native), _ANTHROPIC_FINISH[raw.stop_reason or "end_turn"], usage


def _responses(raw: OpenAIResponse, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage | None]:
    if raw.status == "failed":  # a plain call reports a failed generation as a 200; a stream raises
        raise ProviderError(raw.error.message if raw.error else "the response failed", provider_name="openai")
    # the parse helpers' fields are ours, not openai's: sent back, they'd change the request
    native = [i.model_dump(mode="json", exclude_none=True,
                           exclude={"parsed_arguments": True, "content": {"__all__": {"parsed"}}})
              for i in raw.output]
    parts = [p for i in native if i["type"] == "message" for p in i["content"]]
    text = "".join(p.get("text") or p.get("refusal") or "" for p in parts)
    summary = "\n\n".join(s["text"] for i in native if i["type"] == "reasoning" for s in i.get("summary", []))
    calls = [_call(ids, i["call_id"], i["name"], arguments(i["arguments"]))
             for i in native if i["type"] == "function_call"]
    if raw.status == "incomplete":
        cut = raw.incomplete_details.reason if raw.incomplete_details else None
        finish: Finish = "refusal" if cut == "content_filter" else "length"
    else:
        finish = "calls" if calls else "refusal" if any(p["type"] == "refusal" for p in parts) else "stop"
    u = raw.usage
    usage = Usage(input=u.input_tokens, output=u.output_tokens, reasoning=u.output_tokens_details.reasoning_tokens,
                  cache_read=u.input_tokens_details.cached_tokens,
                  cache_write=u.input_tokens_details.cache_write_tokens) if u else None
    return _message(text, calls, summary, origin, native), finish, usage


def _call(ids: dict[str, str], wire_id: str, name: str, args: dict) -> Call:
    return Call(id=ids.get(wire_id) or new_id(), name=name, arguments=args, wire_id=wire_id)


def _message(text: str, calls: list[Call], reasoning: str | None, origin: Origin, native: Any) -> Message:
    return Message(role="assistant", content=text, calls=calls or None, reasoning=reasoning or None,
                   origin=origin, native=native)
