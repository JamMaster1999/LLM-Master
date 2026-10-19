"""The facade: one held any-llm client per provider, so every call reuses its connection pool."""
from collections.abc import Callable, Sequence
from typing import Literal, get_args, overload

import anthropic
import openai
from any_llm import AnyLLM
from genai_prices import UpdatePrices

from llm_master import fanout, media
from llm_master.io.messages import Message
from llm_master.io.parts import Image
from llm_master.io.response import Response
from llm_master.io.stream import Stream
from llm_master.limiter import Limiter
from llm_master.media import Media
from llm_master.providers import PROVIDERS, SDK, ProviderName
from llm_master.query import run
from llm_master.types import Model

# genai picks aiohttp by itself and closes its own session on garbage collection
_AIOHTTP: dict[SDK, type | None] = {"openai": openai.DefaultAioHttpClient,
                                     "anthropic": anthropic.DefaultAioHttpClient, "genai": None}
assert set(_AIOHTTP) == set(get_args(SDK)), "every SDK needs an aiohttp decision"


class LLM:
    """Hold one per process and aclose it at shutdown. Async-only: sync callers run asyncio.run once."""

    def __init__(self, update_prices: bool = True) -> None:
        """update_prices starts genai-prices' hourly price refresh (SPEC.md Money); off, nothing is fetched."""
        self._clients: dict[ProviderName, AnyLLM] = {}
        self._http: list = []
        self.limiter = Limiter()  # every call through this LLM shares its windows (SPEC.md Rate limits)
        self._prices = UpdatePrices()  # one refresh thread per process, however many LLMs start it
        if update_prices:
            self._prices.start()

    def client(self, provider: ProviderName) -> AnyLLM:
        if provider not in self._clients:
            make = _AIOHTTP[PROVIDERS[provider]["sdk"]]  # KeyError off the roster
            kwargs = {"http_client": make()} if make else {}
            self._http.extend(kwargs.values())
            self._clients[provider] = AnyLLM.create(provider, unified_exceptions=True, max_retries=0, **kwargs)
        return self._clients[provider]

    @overload
    async def query(self, candidates: Model | Sequence[Model], messages: Sequence[Message | dict], *,
                    stream: Literal[False] = False, moderation: bool = False) -> Response: ...
    @overload
    async def query(self, candidates: Model | Sequence[Model], messages: Sequence[Message | dict], *,
                    stream: Literal[True], moderation: bool = False) -> Stream: ...
    @overload
    async def query(self, candidates: Model | Sequence[Model], messages: Sequence[Message | dict], *,
                    stream: bool, moderation: bool = False) -> Response | Stream: ...

    async def query(self, candidates, messages, *, stream=False, moderation=False):
        """The model's answer to messages, from the first candidate that gives one. See IO.md.

        moderation checks the newest user message once, before any candidate (CONFIG.md field 12).
        """
        return await run(self, candidates, messages, stream, moderation)

    async def async_query(self, candidates: Model | Sequence[Model], requests: Sequence[fanout.Request], *,
                          concurrency: int | None = None,
                          progress_callback: Callable[[int, int], None] | None = None
                          ) -> list[Response]:
        """Many query() calls at once, at most concurrency in flight (128 when unset, as in v1); the Responses
        in request order.

        progress_callback(done, total) runs after each request finishes. See SPEC.md Fan-out.
        """
        return await fanout.run(self, candidates, requests, concurrency, progress_callback)

    async def image(self, candidates: Model | Sequence[Model], prompt: str, images: Sequence[Image] = ()) -> Media:
        """A new image, or an edit of the parent images. See SPEC.md Media."""
        return await media.image(self, candidates, prompt, images)

    async def speech(self, candidates: Model | Sequence[Model], text: str) -> Media:
        """The text spoken. See SPEC.md Media."""
        return await media.speech(self, candidates, text)

    async def aclose(self) -> None:
        self._prices.stop()
        http, self._http, self._clients = self._http, [], {}
        for client in http:
            await client.aclose()
