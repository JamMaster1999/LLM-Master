"""The rate limiter: a sliding window per (provider, model), its cap set by the caller. See SPEC.md Rate limits."""
import asyncio
import time
from collections import deque

from llm_master.providers import ProviderName

WINDOW = 60.0  # seconds: the cap counts the tries in any minute


class Limiter:
    """At most per_minute tries start in any WINDOW seconds, per (provider, model); None means no limit."""

    def __init__(self, per_minute: int | None) -> None:
        if per_minute is not None and per_minute < 1:  # 0 would hold every try forever
            raise ValueError(f"per_minute is 1 or more, or None for no limit, not {per_minute}")
        self._cap = per_minute
        self._starts: dict[tuple[ProviderName, str], deque[float]] = {}

    async def wait(self, provider: ProviderName, model: str) -> None:
        """Returns once a try may start, and counts it as started."""
        if self._cap is None:
            return
        starts = self._starts.setdefault((provider, model), deque())
        while True:
            now = time.monotonic()
            while starts and starts[0] <= now - WINDOW:
                starts.popleft()
            if len(starts) < self._cap:
                starts.append(now)
                return
            await asyncio.sleep(starts[0] + WINDOW - now)  # until the oldest start leaves the window
