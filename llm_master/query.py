"""query(): one conversation call. Builds each candidate's request, sends it, reads the Response. See IO.md."""
import time
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, assert_never

from any_llm import AnyLLM

from llm_master.config import (cache, max_tokens, merge, moderation, response_format, stream as streaming, system,
                               temperature, thinking, timeout, tool_choice, tools)
from llm_master.io.messages import Message, Origin, assemble, validate
from llm_master.io.parts import Audio, Text
from llm_master.io.response import Attempt, Response, parse, read
from llm_master.io.stream import Stream, chunks
from llm_master.providers import API, PROVIDERS
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM


def build(candidate: Model, history: list[Message], stream: bool) -> tuple[API, dict]:
    """The API and kwargs one candidate gets. Pure, like assembly: same input, same request."""
    api = _api(candidate, history)
    c = candidate.config
    fragments = [assemble(history, candidate.provider, candidate.model, api, c.system, breakpoint=isinstance(c.cache, int)),
                 streaming.translate(stream, api), max_tokens.translate(c.max_tokens, api), cache.translate(c.cache, api)]
    if c.system is not None:
        fragments.append(system.translate(c.system, api))
    if c.temperature is not None:
        fragments.append(temperature.translate(c.temperature, api))
    if c.thinking is not None:
        fragments.append(thinking.translate(c.thinking, api))
    if c.tools is not None:
        fragments.append(tools.translate(c.tools, api))
    if c.tool_choice is not None:
        fragments.append(tool_choice.translate(c.tool_choice, api))
    if c.response_format is not None:
        fragments.append(response_format.translate(c.response_format, api))
    if c.timeout is not None:  # after max_tokens, so it beats anthropic's default 600
        fragments.append(timeout.translate(c.timeout, api))
    return api, merge(*fragments, candidate.extra)


async def run(llm: "LLM", candidates: Model | Sequence[Model], messages: Sequence[Message | dict],
              stream: bool) -> Response | Stream:
    candidate = candidates if isinstance(candidates, Model) else candidates[0]  # fallback: SPEC.md step 7
    history = validate(messages)
    if candidate.config.moderation:
        await moderation.check(llm.client("openai"), _last_user_text(history))
    started = time.monotonic()
    api, request = build(candidate, history, stream)
    if stream:
        return Stream(_streamed(llm, candidate, api, request, started))
    raw = await _send(llm.client(candidate.provider), api, candidate.model, request)
    message, finish, usage = read(api, raw, _origin(candidate), {})
    return _response(candidate, message, finish, usage, started)


async def _streamed(llm: "LLM", candidate: Model, api: API, request: dict,
                    started: float) -> AsyncGenerator[dict | Response, None]:
    ids: dict[str, str] = {}
    final = None
    async with aclosing(await _send(llm.client(candidate.provider), api, candidate.model, request)) as events:
        yield {"type": "start-step"}
        try:
            async for item in chunks(api, events, ids):
                if isinstance(item, dict):
                    yield item
                else:
                    final = item
        except Exception:
            yield {"type": "reset-step"}  # the browser drops this attempt's output
            raise
    yield {"type": "finish-step"}
    message, finish, usage = read(api, final, _origin(candidate), ids)
    yield _response(candidate, message, finish, usage, started)


async def _send(client: AnyLLM, api: API, model: str, request: dict) -> Any:
    match api:
        case "completion":
            return await client.acompletion(model=model, **request)
        case "messages":
            return await client.amessages(model=model, **request)
        case "responses":
            return await client.aresponses(model=model, **request)
        case _:
            assert_never(api)


def _api(candidate: Model, history: list[Message]) -> API:
    """The provider's API, or its audio_api when the history holds audio (openai's responses takes none)."""
    row = PROVIDERS[candidate.provider]
    audio = any(isinstance(p, Audio) for m in history if isinstance(m.content, list) for p in m.content)
    return row.get("audio_api", row["api"]) if audio else row["api"]


def _response(candidate: Model, message: Message, finish, usage, started: float) -> Response:
    latency = time.monotonic() - started
    return Response(message=message, parsed=parse(candidate.config.response_format, message, finish), finish=finish,
                    usage=usage, attempts=[Attempt(candidate.provider, candidate.model, latency)], latency=latency)


def _origin(candidate: Model) -> Origin:
    return Origin(provider=candidate.provider, model=candidate.model)


def _last_user_text(history: list[Message]) -> str:
    """What moderation reads: the newest user message's text, as v1 did."""
    content = next((m.content for m in reversed(history) if m.role == "user"), "")
    return content if isinstance(content, str) else "\n".join(p.text for p in content if isinstance(p, Text))
