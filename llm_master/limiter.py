"""The local rate limiter: a sliding window per (provider, model). See SPEC.md Rate limits."""
import asyncio
import time
from collections import deque

from llm_master.providers import ProviderName
from llm_master.registry import limit


class Limiter:
    """At most `requests` tries start in any `interval` seconds, per (provider, model)."""

    def __init__(self) -> None:
        self._starts: dict[tuple[ProviderName, str], deque[float]] = {}

    async def wait(self, provider: ProviderName, model: str) -> None:
        """Returns once a try may start, and counts it as started."""
        requests, interval = limit(provider, model)
        starts = self._starts.setdefault((provider, model), deque())
        while True:
            now = time.monotonic()
            while starts and starts[0] <= now - interval:
                starts.popleft()
            if len(starts) < requests:
                starts.append(now)
                return
            await asyncio.sleep(starts[0] + interval - now)  # until the oldest start leaves the window
