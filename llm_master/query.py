"""query(): one conversation call. Builds each candidate's request, sends it, reads the Response. See IO.md."""
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, assert_never

from any_llm import AnyLLM

from llm_master import pipeline
from llm_master.config import (cache, max_tokens, merge, moderation as moderating, response_format,
                               stream as streaming, system, temperature, thinking, timeout, tool_choice, tools)
from llm_master.io.messages import Message, Origin, assemble, validate
from llm_master.io.parts import Audio
from llm_master.io.response import Response, read
from llm_master.io.stream import Stream, adapter
from llm_master.providers import API, PROVIDERS
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM


def build(candidate: Model, history: list[Message], stream: bool) -> tuple[API, dict]:
    """The API and kwargs one candidate gets. Pure, like assembly: same input, same request."""
    api = _api(candidate, history)
    c = candidate.config
    # every field, unset ones too: each translate says what unset means on each API
    fragments = [assemble(history, candidate.provider, candidate.model, api, c.system),
                 system.translate(c.system, api), streaming.translate(stream, api),
                 temperature.translate(c.temperature, api), thinking.translate(c.thinking, api),
                 tools.translate(c.tools, api), tool_choice.translate(c.tool_choice, api),
                 response_format.translate(c.response_format, api), max_tokens.translate(c.max_tokens, api),
                 timeout.translate(c.timeout, api), cache.translate(c.cache, api)]
    return api, merge(*fragments, candidate.extra)


async def run(llm: "LLM", candidates: Model | Sequence[Model], messages: Sequence[Message | dict],
              stream: bool, moderation: bool) -> Response | Stream:
    history = validate(messages)
    chain = pipeline.chain(candidates)
    if moderation:
        await moderating.check(llm.client("openai"), history)
    if stream:
        return Stream(pipeline.stream(chain, lambda c: _streamed(llm, c, history)))

    async def once(candidate: Model) -> Response:
        api, request = build(candidate, history, stream=False)
        raw = await _send(llm, candidate, api, request)
        return read(api, raw, _origin(candidate), {}, candidate.config.response_format)

    return await pipeline.run(chain, once)


async def _streamed(llm: "LLM", candidate: Model, history: list[Message]
                    ) -> AsyncGenerator[dict | pipeline.Final[Response], None]:
    """One streamed try: start-step, its chunks, finish-step, then its Response. Each try marks its own steps."""
    api, request = build(candidate, history, stream=True)
    convert, opened = adapter(api), False
    try:
        async with aclosing(await _send(llm, candidate, api, request)) as events:
            async for event in events:
                if not opened:  # the request opened: the provider's first event arrived
                    opened = True
                    yield {"type": "start-step"}
                for chunk in convert.feed(event):
                    yield chunk
        tail, raw = convert.finish()
        for chunk in tail:
            yield chunk
        response = read(api, raw, _origin(candidate), convert.tools.ids, candidate.config.response_format)
    except Exception:
        if opened:
            yield {"type": "reset-step"}  # the browser drops this try's output
        raise
    yield {"type": "finish-step"}
    yield pipeline.Final(response)


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
