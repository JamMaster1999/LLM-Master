"""query(): one conversation call. Builds each candidate's request, sends it, reads the Response. See IO.md."""
import time
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, assert_never

from any_llm import AnyLLM

from llm_master import pipeline
from llm_master.config import (cache, max_tokens, merge, moderation, response_format, stream as streaming, system,
                               temperature, thinking, timeout, tool_choice, tools)
from llm_master.io.messages import Message, Origin, assemble, validate
from llm_master.io.parts import Audio, Text
from llm_master.io.response import Attempt, Response, parse, read
from llm_master.io.stream import Stream, adapter
from llm_master.providers import API, PROVIDERS
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM


def build(candidate: Model, history: list[Message], stream: bool) -> tuple[API, dict]:
    """The API and kwargs one candidate gets. Pure, like assembly: same input, same request."""
    api = _api(candidate, history)
    c = candidate.config
    fragments = [assemble(history, candidate.provider, candidate.model, api, c.system,
                          cache_breakpoint=isinstance(c.cache, int)),
                 streaming.translate(stream, api), max_tokens.translate(c.max_tokens, api),
                 timeout.translate(c.timeout, api), cache.translate(c.cache, api)]
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
    return api, merge(*fragments, candidate.extra)


async def run(llm: "LLM", candidates: Model | Sequence[Model], messages: Sequence[Message | dict],
              stream: bool) -> Response | Stream:
    history = validate(messages)
    chain = pipeline.chain(candidates)
    if any(c.config.moderation for c in chain):  # once per call: the input is the same for every candidate
        await moderation.check(llm.client("openai"), _last_user_text(history))
    if stream:
        api, request = build(chain[0], history, stream=True)  # a field with no spelling raises before the stream
        return Stream(_streamed(llm, chain[0], api, request))

    async def once(candidate: Model):
        api, request = build(candidate, history, stream=False)
        message, finish, usage = read(api, await _send(llm, candidate, api, request), _origin(candidate), {})
        return message, finish, usage, parse(candidate.config.response_format, message, finish)

    (message, finish, usage, parsed), _, attempts, latency = await pipeline.run(chain, once)
    return Response(message=message, parsed=parsed, finish=finish, usage=usage, attempts=attempts, latency=latency)


async def _streamed(llm: "LLM", candidate: Model, api: API, request: dict) -> AsyncGenerator[dict | Response, None]:
    started, ids, opened = time.monotonic(), {}, False
    convert = adapter(api, ids)
    async with aclosing(await _send(llm, candidate, api, request)) as events:
        try:
            async for event in events:
                if not opened:  # the request opened: the provider's first event arrived
                    opened = True
                    yield {"type": "start-step"}
                for chunk in convert.feed(event):
                    yield chunk
            tail, raw = convert.finish()
            for chunk in tail:
                yield chunk
            message, finish, usage = read(api, raw, _origin(candidate), ids)
            parsed = parse(candidate.config.response_format, message, finish)
        except Exception:
            if opened:
                yield {"type": "reset-step"}  # the browser drops this attempt's output
            raise
    yield {"type": "finish-step"}
    latency = time.monotonic() - started
    yield Response(message=message, parsed=parsed, finish=finish, usage=usage, latency=latency,
                   attempts=[Attempt(candidate.provider, candidate.model, latency)])


async def _send(llm: "LLM", candidate: Model, api: API, request: dict) -> Any:
    client: AnyLLM = llm.client(candidate.provider)
    match api:
        case "completion":
            return await client.acompletion(model=candidate.model, **request)
        case "messages":
            return await client.amessages(model=candidate.model, **request)
        case "responses":
            return await client.aresponses(model=candidate.model, **request)
        case _:
            assert_never(api)


def _api(candidate: Model, history: list[Message]) -> API:
    """The provider's API, or its audio_api when the history holds audio (openai's responses takes none)."""
    row = PROVIDERS[candidate.provider]
    audio = any(isinstance(p, Audio) for m in history if isinstance(m.content, list) for p in m.content)
    return row.get("audio_api", row["api"]) if audio else row["api"]


def _origin(candidate: Model) -> Origin:
    return Origin(provider=candidate.provider, model=candidate.model)


def _last_user_text(history: list[Message]) -> str:
    """What moderation reads: the newest user message's text, as v1 did."""
    content = next((m.content for m in reversed(history) if m.role == "user"), "")
    return content if isinstance(content, str) else "\n".join(p.text for p in content if isinstance(p, Text))
