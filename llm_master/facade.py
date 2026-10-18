"""The facade: one held any-llm client per provider, so every call reuses its connection pool."""
from any_llm import AnyLLM

from llm_master.providers import PROVIDERS, ProviderName


class LLM:
    """Hold one per process. Async-only: sync callers run asyncio.run once, since pools bind to a loop."""

    def __init__(self) -> None:
        self._clients: dict[ProviderName, AnyLLM] = {}

    def client(self, provider: ProviderName) -> AnyLLM:
        """The provider's held client, made on first use: env credentials, typed errors, no SDK retries."""
        if provider not in self._clients:
            if provider not in PROVIDERS:
                raise KeyError(provider)  # unknown provider fails loud, as in registry.py
            self._clients[provider] = AnyLLM.create(provider, unified_exceptions=True, max_retries=0)
        return self._clients[provider]
