"""The Response a call returns, read from each API's output. See IO.md."""
import json
from dataclasses import dataclass, field
from typing import Any, Literal, assert_never, cast

from any_llm.exceptions import InvalidRequestError, ProviderError
from any_llm.providers.anthropic.utils import ANTHROPIC_STOP_REASON_TO_FINISH_REASON
from any_llm.types.completion import ChatCompletion, CompletionUsage
from any_llm.types.messages import MessageResponse
from openai.types.responses import Response as OpenAIResponse
from pydantic import TypeAdapter

from llm_master.config.response_format import Format
from llm_master.io.messages import Call, Message, Origin, new_id
from llm_master.providers import API, ProviderName

Finish = Literal["stop", "length", "tool_calls", "content_filter"]  # any-llm's words
_PASSING = {"server_error", "rate_limit_exceeded", "vector_store_timeout"}  # openai failures worth a retry


@dataclass
class Usage:
    """input counts every prompt token, cache reads and writes included; output counts reasoning.

    The rest are parts of those two that providers price apart (SPEC.md Money).
    """

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0  # anthropic's 1-hour writes, part of cache_write
    input_audio: int = 0
    input_image: int = 0
    output_audio: int = 0
    output_image: int = 0


@dataclass
class Attempt:
    """One candidate tried, with one typed error per failed try."""

    provider: ProviderName
    model: str
    latency: float
    errors: list[Exception] = field(default_factory=list)


@dataclass
class Response:
    """message goes into history; everything else describes this one call."""

    message: Message
    parsed: Any
    finish: Finish
    usage: Usage | None  # None when the provider reported none: unknown, never zero
    attempts: list[Attempt] = field(default_factory=list)  # the pipeline sets these three
    latency: float = 0.0
    cost: float | None = None  # dollars; None when unknown (SPEC.md Money)


class CutCall(ProviderError):
    """A tool call the model didn't finish, cut off by max_tokens or a refusal: it can't run."""


def read(api: API, raw: Any, origin: Origin, ids: dict[str, str], fmt: Format | None) -> Response:
    """One API's output as a Response. ids holds the wire ids a stream already gave our ids."""
    match api:
        case "completion":
            message, finish, usage = _completion(raw, origin, ids)
        case "messages":
            message, finish, usage = _messages(raw, origin, ids)
        case "responses":
            message, finish, usage = _responses(raw, origin, ids)
        case _:
            assert_never(api)
    if message.calls and finish in ("length", "content_filter"):  # the last call stopped mid-arguments
        raise CutCall(f"the answer stopped ({finish}) inside its tool calls, so they can't run")
    return Response(message=message, parsed=_parse(fmt, message, finish), finish=finish, usage=usage)


def arguments(text: str) -> dict:
    """A call's arguments: one whole JSON object, or CutCall. A call with no arguments sends "{}"."""
    try:
        args = json.loads(text)
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        raise CutCall(f"a tool call's arguments aren't one whole JSON object, so it can't run: {text[:80]!r}")
    return args


def completion_usage(u: CompletionUsage | None) -> Usage | None:
    """any-llm's chat usage; gemini's media calls report it too."""
    if u is None:
        return None
    sent, made = u.prompt_tokens_details, u.completion_tokens_details
    return Usage(input=u.prompt_tokens, output=u.completion_tokens, reasoning=made and made.reasoning_tokens or 0,
                 cache_read=sent and sent.cached_tokens or 0, cache_write=sent and sent.cache_write_tokens or 0,
                 input_audio=sent and sent.audio_tokens or 0, input_image=sent and sent.image_tokens or 0,
                 output_audio=made and made.audio_tokens or 0, output_image=made and made.image_tokens or 0)


def _completion(raw: ChatCompletion, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage | None]:
    if not raw.choices:  # gemini's MALFORMED_FUNCTION_CALL and UNEXPECTED_TOOL_CALL leave no choice at all
        raise ProviderError("the model answered with no choice", provider_name=origin.provider)
    choice = raw.choices[0]
    native = choice.message.model_dump(mode="json", exclude_none=True)
    calls = [_call(ids, tc["id"], tc["function"]["name"], arguments(tc["function"]["arguments"]))
             for tc in native.get("tool_calls", [])]
    text = native.get("content") or native.get("refusal") or ""
    # openai reports a refusal as a normal stop; only the refusal field tells. function_call never
    # comes back: it answers the legacy functions parameter, which llm_master never sends
    finish = "content_filter" if native.get("refusal") else cast(Finish, choice.finish_reason)
    return _message(text, calls, native.get("reasoning"), origin, native), finish, completion_usage(raw.usage)


def _messages(raw: MessageResponse, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage]:
    native = [b.model_dump(mode="json", exclude_none=True) for b in raw.content]
    text = "".join(b["text"] for b in native if b["type"] == "text")
    thinking = "\n\n".join(b["thinking"] for b in native if b["type"] == "thinking")
    calls = [_call(ids, b["id"], b["name"], b["input"]) for b in native if b["type"] == "tool_use"]
    u = raw.usage
    cache_read, cache_write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
    usage = Usage(input=u.input_tokens + cache_read + cache_write, output=u.output_tokens,
                  reasoning=u.output_tokens_details.thinking_tokens if u.output_tokens_details else 0,
                  cache_read=cache_read, cache_write=cache_write,
                  cache_write_1h=u.cache_creation.ephemeral_1h_input_tokens if u.cache_creation else 0)
    finish = ANTHROPIC_STOP_REASON_TO_FINISH_REASON.get(raw.stop_reason or "end_turn", "stop")
    return _message(text, calls, thinking, origin, native), finish, usage


def _responses(raw: OpenAIResponse, origin: Origin, ids: dict[str, str]) -> tuple[Message, Finish, Usage | None]:
    if raw.status == "failed":  # a failed generation still answers 200, plain or streamed
        error = raw.error
        if error and error.code not in _PASSING:  # invalid_prompt, bio_policy, a bad image: it fails again
            raise InvalidRequestError(error.message, provider_name="openai", code=error.code)
        raise ProviderError(error.message if error else "the response failed", provider_name="openai",
                            code=error.code if error else None)
    native = [i.model_dump(mode="json", exclude_none=True) for i in raw.output]
    parts = [p for i in native if i["type"] == "message" for p in i["content"]]
    text = "".join(p.get("text") or p.get("refusal") or "" for p in parts)
    summary = "\n\n".join(s["text"] for i in native if i["type"] == "reasoning" for s in i.get("summary", []))
    calls = [_call(ids, i["call_id"], i["name"], arguments(i["arguments"]))
             for i in native if i["type"] == "function_call"]
    if raw.status == "incomplete":
        cut = raw.incomplete_details.reason if raw.incomplete_details else None
        finish: Finish = "content_filter" if cut == "content_filter" else "length"
    else:
        finish = "tool_calls" if calls else "content_filter" if any(p["type"] == "refusal" for p in parts) else "stop"
    u = raw.usage
    usage = Usage(input=u.input_tokens, output=u.output_tokens, reasoning=u.output_tokens_details.reasoning_tokens,
                  cache_read=u.input_tokens_details.cached_tokens,
                  cache_write=u.input_tokens_details.cache_write_tokens) if u else None
    return _message(text, calls, summary, origin, native), finish, usage


def _parse(fmt: Format | None, message: Message, finish: Finish) -> Any:
    """The response_format object, or None when there's none to give: the caller decides what that means."""
    if fmt is None or fmt.type == "text" or finish != "stop" or not isinstance(message.content, str):
        return None
    try:
        if isinstance(fmt.schema_, type):
            return TypeAdapter(fmt.schema_).validate_json(message.content)
        return json.loads(message.content)
    except ValueError:  # pydantic's ValidationError and json's JSONDecodeError are both ValueErrors
        return None


def _call(ids: dict[str, str], wire_id: str, name: str, args: dict) -> Call:
    return Call(id=ids.get(wire_id) or new_id(), name=name, arguments=args, wire_id=wire_id)


def _message(text: str, calls: list[Call], reasoning: str | None, origin: Origin, native: Any) -> Message:
    return Message(role="assistant", content=text, calls=calls or None, reasoning=reasoning or None,
                   origin=origin, native=native)
