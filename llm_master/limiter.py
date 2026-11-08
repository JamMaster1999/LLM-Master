"""The rate limiter: a sliding window per (provider, model), its cap set by the caller. See SPEC.md Rate limits."""
import asyncio
import time
from collections import defaultdict, deque

from llm_master.providers import ProviderName

WINDOW = 60.0  # seconds: the cap counts the tries in any minute


class Limiter:
    """At most per_minute tries start in any WINDOW seconds, per (provider, model); None means no limit."""

    def __init__(self, per_minute: int | None) -> None:
        if per_minute is not None and per_minute < 1:  # a cap of 0 would let no try start
            raise ValueError(f"per_minute is 1 or more, or None for no limit, not {per_minute}")
        self._cap = per_minute
        # the last per_minute starts: a try may start once the oldest of them is a window old
        self._windows: defaultdict[tuple[ProviderName, str], tuple[asyncio.Lock, deque[float]]] = defaultdict(
            lambda: (asyncio.Lock(), deque(maxlen=per_minute)))

    async def wait(self, provider: ProviderName, model: str) -> None:
        """Returns once a try may start, and counts it as started. Waiters go in arrival order."""
        if self._cap is None:
            return
        lock, starts = self._windows[provider, model]
        async with lock:
            while len(starts) == self._cap and (left := starts[0] + WINDOW - time.monotonic()) > 0:
                await asyncio.sleep(left)
            starts.append(time.monotonic())  # maxlen drops the oldest start, now out of the window
