"""The facade: one held any-llm client per provider, so every call reuses its connection pool."""
from typing import get_args

import anthropic
import openai
from any_llm import AnyLLM

from llm_master.providers import PROVIDERS, SDK, ProviderName

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

    async def aclose(self) -> None:
        http, self._http, self._clients = self._http, [], {}
        for client in http:
            await client.aclose()
