"""The facade: one held any-llm client per provider, so every call reuses its connection pool."""
from collections.abc import Sequence
from typing import Literal, get_args, overload

import anthropic
import openai
from any_llm import AnyLLM

from llm_master.io.messages import Message
from llm_master.io.response import Response
from llm_master.io.stream import Stream
from llm_master.providers import PROVIDERS, SDK, ProviderName
from llm_master.query import run
from llm_master.types import Model

# genai picks aiohttp by itself and closes its own session on garbage collection
_AIOHTTP: dict[SDK, type | None] = {"openai": openai.DefaultAioHttpClient,
                                     "anthropic": anthropic.DefaultAioHttpClient, "genai": None}
assert set(_AIOHTTP) == set(get_args(SDK)), "every SDK needs an aiohttp decision"


class LLM:
    """Hold one per process and aclose it at shutdown. Async-only: sync callers run asyncio.run once."""

    def __init__(self) -> None:
        self._clients: dict[ProviderName, AnyLLM] = {}
        self._http: list = []

    def client(self, provider: ProviderName) -> AnyLLM:
        if provider not in self._clients:
            make = _AIOHTTP[PROVIDERS[provider]["sdk"]]  # KeyError off the roster
            kwargs = {"http_client": make()} if make else {}
            self._http.extend(kwargs.values())
            self._clients[provider] = AnyLLM.create(provider, unified_exceptions=True, max_retries=0, **kwargs)
        return self._clients[provider]

    @overload
    async def query(self, candidates: Model | Sequence[Model], messages: Sequence[Message | dict], *,
                    stream: Literal[False] = False) -> Response: ...
    @overload
    async def query(self, candidates: Model | Sequence[Model], messages: Sequence[Message | dict], *,
                    stream: Literal[True]) -> Stream: ...
    @overload
    async def query(self, candidates: Model | Sequence[Model], messages: Sequence[Message | dict], *,
                    stream: bool) -> Response | Stream: ...

    async def query(self, candidates, messages, *, stream=False):
        """The model's answer to messages, from the first candidate that gives one. See IO.md."""
        return await run(self, candidates, messages, stream)

    async def aclose(self) -> None:
        http, self._http, self._clients = self._http, [], {}
        for client in http:
            await client.aclose()
