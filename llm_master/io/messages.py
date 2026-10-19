"""llm_master's message format, its validator, and its spelling per API. See IO.md."""
import json
import secrets
from collections.abc import Callable, Sequence
from typing import Any, Literal, assert_never

from pydantic import BaseModel, Field, JsonValue, model_validator

from llm_master.config import UnsupportedField
from llm_master.io.parts import Audio, File, Image, Part, Text, uri
from llm_master.providers import API, ProviderName


def new_id() -> str:
    return "lm_" + secrets.token_hex(12)


class Call(BaseModel, extra="forbid"):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")  # ours, see IO.md Ids
    name: str
    arguments: dict[str, JsonValue]
    wire_id: str | None = None  # the provider's id, sent only when native replays


class Origin(BaseModel, extra="forbid"):
    provider: ProviderName
    model: str


class Message(BaseModel, extra="forbid"):
    """One message of any role; role decides which fields apply."""

    role: Literal["user", "assistant", "tool"]
    content: str | list[Part] = ""
    call_id: str | None = None   # tool only: the call it answers
    calls: list[Call] | None = None
    reasoning: str | None = None
    origin: Origin | None = None
    native: Any = None           # the provider's raw output, JSON, never re-validated

    @model_validator(mode="after")
    def _fields_fit_role(self):
        assistant_only = {"calls": self.calls, "reasoning": self.reasoning, "origin": self.origin,
                          "native": self.native}
        if self.role != "assistant" and (misplaced := [k for k, v in assistant_only.items() if v is not None]):
            raise ValueError(f"{self.role} messages can't carry {', '.join(misplaced)}")
        if (self.role == "tool") != (self.call_id is not None):
            raise ValueError("call_id is required on tool messages and forbidden on the others")
        match self.role:
            case "user":
                pass
            case "tool":
                if isinstance(self.content, list) and any(isinstance(p, Audio) for p in self.content):
                    raise ValueError("tool results hold text, image and file parts; audio is user-only")
            case "assistant":
                if isinstance(self.content, list):
                    raise ValueError("assistant content is a string")
                if self.native is not None and self.origin is None:
                    raise ValueError("native requires origin")
                if not self.content and not self.calls and self.native is None:
                    raise ValueError("an assistant message needs content, calls or native")
            case _:
                assert_never(self.role)
        return self


def validate(messages: Sequence[Message | dict]) -> list[Message]:
    """Every rule of IO.md Validation; raises ValueError (pydantic's for a bad message) before anything is sent."""
    history = [Message.model_validate(m) for m in messages]
    if not history:
        raise ValueError("messages can't be empty")
    seen: set[str] = set()
    open_calls: set[str] = set()
    for m in history:
        if m.role == "tool":
            if m.call_id not in open_calls:
                raise ValueError(f"tool result {m.call_id!r} answers no open call of the assistant message before it")
            open_calls.remove(m.call_id)
            continue
        if open_calls:
            raise ValueError(f"calls {sorted(open_calls)} are unanswered before the next {m.role} message")
        for call in m.calls or []:
            if call.id in seen:
                raise ValueError(f"call id {call.id!r} repeats")
            seen.add(call.id)
        open_calls = {call.id for call in m.calls or []}
    if open_calls:
        raise ValueError(f"calls {sorted(open_calls)} are unanswered")
    return history


def assemble(messages: list[Message], provider: ProviderName, model: str, api: API, system: str | None) -> dict:
    """The history spelled for one candidate. Pure: same history and candidate, same bytes.

    system rides here because completion has no parameter for it.
    """
    me = Origin(provider=provider, model=model)
    replays = [m.native is not None and m.origin == me for m in messages]
    wire = {}  # our call id → the provider's, for the calls that replay natively
    for m, replay in zip(messages, replays):
        if replay:
            wire |= {c.id: c.wire_id for c in m.calls or [] if c.wire_id}
    match api:
        case "completion":
            return {"messages": _completion(messages, replays, wire, system)}
        case "messages":
            return {"messages": _messages(messages, replays, wire)}
        case "responses":
            # saved history is the only conversation state, and reasoning replays without openai's servers
            return {"input_data": _responses(messages, replays, wire), "store": False,
                    "include": ["reasoning.encrypted_content"]}
        case _:
            assert_never(api)


def _completion(messages: list[Message], replays: list[bool], wire: dict, system: str | None) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}] if system is not None else []
    for m, replay in zip(messages, replays):
        match m.role:
            case "user":
                out.append({"role": "user", "content": _each(m.content, _completion_part)})
            case "assistant" if replay:
                out.append(m.native)
            case "assistant" if not m.content and not m.calls:
                pass  # another model's reasoning-only turn: nothing readable to send
            case "assistant":
                message: dict = {"role": "assistant", "content": m.content or None}
                if m.calls:
                    message["tool_calls"] = [{"id": c.id, "type": "function",
                                              "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                             for c in m.calls]
                out.append(message)
            case "tool":
                out.append({"role": "tool", "tool_call_id": wire.get(m.call_id, m.call_id),
                            "content": _each(m.content, _completion_part)})
            case _:
                assert_never(m.role)
    return out


def _messages(messages: list[Message], replays: list[bool], wire: dict) -> list[dict]:
    out: list[dict] = []
    for i, (m, replay) in enumerate(zip(messages, replays)):
        match m.role:
            case "user":
                out.append({"role": "user", "content": _each(m.content, _messages_part)})
            case "assistant" if replay and m.native:  # claude's empty answer (content: []) 400s when sent back
                out.append({"role": "assistant", "content": m.native})
            case "assistant":
                blank = not isinstance(m.content, str) or not m.content.strip()
                blocks = [] if blank else [{"type": "text", "text": m.content}]  # a blank text block 400s
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.calls or []]
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
            case "tool":
                result: dict = {"type": "tool_result", "tool_use_id": wire.get(m.call_id, m.call_id)}
                if m.content:
                    result["content"] = _each(m.content, _messages_part)
                if i and messages[i - 1].role == "tool":  # consecutive results share one user message
                    out[-1]["content"].append(result)
                else:
                    out.append({"role": "user", "content": [result]})
            case _:
                assert_never(m.role)
    return out


def _responses(messages: list[Message], replays: list[bool], wire: dict) -> list[dict]:
    out: list[dict] = []
    for m, replay in zip(messages, replays):
        match m.role:
            case "user":
                out.append({"role": "user", "content": _each(m.content, _responses_part)})
            case "assistant" if replay:
                items = m.native
                if items and items[-1].get("type") == "reasoning":  # truncated: its required next item is missing
                    items = items[:-1]
                out.extend(items)
            case "assistant":
                if m.content:
                    out.append({"role": "assistant", "content": m.content})
                # no item id: an fc_ id without its reasoning item 400s
                out.extend({"type": "function_call", "call_id": c.id, "name": c.name,
                            "arguments": json.dumps(c.arguments)} for c in m.calls or [])
            case "tool":
                out.append({"type": "function_call_output", "call_id": wire.get(m.call_id, m.call_id),
                            "output": _each(m.content, _responses_part)})
            case _:
                assert_never(m.role)
    return out


def _each(content: str | list[Part], spell: Callable[[Part], dict]) -> str | list[dict]:
    return content if isinstance(content, str) else [spell(p) for p in content]


def _completion_part(p: Part) -> dict:
    match p:
        case Text():
            return {"type": "text", "text": p.text}
        case Image():
            return {"type": "image_url", "image_url": {"url": uri(p.mime, p.data)}}
        case File():
            return {"type": "file", "file": {"file_data": uri(p.mime, p.data), "filename": p.filename}}
        case Audio():
            return {"type": "input_audio", "input_audio": {"data": p.data, "format": p.format}}
        case _:
            assert_never(p)


def _messages_part(p: Part) -> dict:
    match p:
        case Text():
            return {"type": "text", "text": p.text}
        case Image():
            return {"type": "image", "source": {"type": "base64", "media_type": p.mime, "data": p.data}}
        case File():
            return {"type": "document", "source": {"type": "base64", "media_type": p.mime, "data": p.data}}
        case Audio():
            raise UnsupportedField("messages", "messages", "anthropic has no audio input")
        case _:
            assert_never(p)


def _responses_part(p: Part) -> dict:
    match p:
        case Text():
            return {"type": "input_text", "text": p.text}
        case Image():
            return {"type": "input_image", "image_url": uri(p.mime, p.data)}
        case File():
            return {"type": "input_file", "file_data": uri(p.mime, p.data), "filename": p.filename}
        case Audio():
            raise UnsupportedField("responses", "messages",
                                   "responses takes no audio input; openai audio uses completion")
        case _:
            assert_never(p)
