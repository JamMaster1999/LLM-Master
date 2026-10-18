"""The facade: one held any-llm client per provider, so every call reuses its connection pool."""
from typing import TYPE_CHECKING, Any, assert_never, cast

import anthropic
import openai
from any_llm import AnyLLM

from llm_master.providers import PROVIDERS, ProviderName

if TYPE_CHECKING:
    from any_llm.providers.anthropic import AnthropicProvider
    from any_llm.providers.fireworks import FireworksProvider
    from any_llm.providers.gemini import GeminiProvider
    from any_llm.providers.openai import OpenaiProvider
    from any_llm.providers.vertexai import VertexaiProvider


class LLM:
    """Hold one per process and aclose it at shutdown. Async-only: sync callers run asyncio.run once,
    since pools bind to a loop."""

    def __init__(self) -> None:
        self._clients: dict[ProviderName, AnyLLM] = {}

    def client(self, provider: ProviderName) -> AnyLLM:
        """The provider's held client, made on first use: env credentials, typed errors, no SDK retries."""
        if provider not in self._clients:
            if provider not in PROVIDERS:
                raise KeyError(provider)  # unknown provider fails loud, as in registry.py
            self._clients[provider] = AnyLLM.create(provider, unified_exceptions=True, max_retries=0,
                                                    **_transport(provider))
        return self._clients[provider]

    async def aclose(self) -> None:
        """Close and forget every held client; a later call makes fresh ones."""
        clients, self._clients = self._clients, {}
        for provider, llm in clients.items():
            await _close(provider, llm)


def _transport(provider: ProviderName) -> dict[str, Any]:
    """Client kwargs that put the provider's SDK on aiohttp."""
    match provider:
        case "openai":
            return {"http_client": openai.DefaultAioHttpClient()}
        case "anthropic":
            return {"http_client": anthropic.DefaultAioHttpClient()}
        case "gemini":
            return {}  # genai picks aiohttp by itself whenever it is installed
        case "vertexai":
            return {}  # the same genai client as gemini
        case "fireworks":
            return {"http_client": openai.DefaultAioHttpClient()}
        case _:
            assert_never(provider)


async def _close(provider: ProviderName, llm: AnyLLM) -> None:
    """Close the SDK client under the provider's any-llm client; any-llm has no close of its own."""
    match provider:
        case "openai":
            await cast("OpenaiProvider", llm).client.close()
        case "anthropic":
            await cast("AnthropicProvider", llm).client.close()
        case "gemini":
            await cast("GeminiProvider", llm).client.aio.aclose()
        case "vertexai":
            await cast("VertexaiProvider", llm).client.aio.aclose()
        case "fireworks":
            await cast("FireworksProvider", llm).client.close()
        case _:
            assert_never(provider)
