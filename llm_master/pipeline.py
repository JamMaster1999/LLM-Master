"""The pipeline every call runs through: query(), image() and speech() hand it one try per candidate.

A try is an async generator: a streamed try yields its chunks, and every try yields its Final last. The
pipeline stamps every attempt, the whole call's latency and its cost on that CallReturn. Per try, the rate
limiter first, then the try, then retries (SPEC.md Failures). Step 7 adds fallback over the candidates.
"""
import asyncio
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from any_llm.exceptions import GatewayTimeoutError, ProviderError, RateLimitError, UpstreamProviderError

from llm_master.cost import price
from llm_master.io.response import Attempt, CutCall, Usage
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM


class CallReturn(Protocol):
    """What a call returns, a Response or a Media: the pipeline prices its usage and sets the rest."""

    usage: Usage | None
    attempts: list[Attempt]
    latency: float
    cost: float | None


A = TypeVar("A", bound=CallReturn)
TRIES = 5  # per candidate


@dataclass
class Final(Generic[A]):
    """A try's last item, its CallReturn. The label keeps it from ever passing for a chunk."""

    value: A


def chain(candidates: Model | Sequence[Model]) -> list[Model]:
    chain = [candidates] if isinstance(candidates, Model) else list(candidates)
    if len(chain) != 1:  # never drop a fallback silently: it lands in SPEC.md step 7
        raise NotImplementedError(f"{len(chain)} candidates: a call takes exactly one until fallback (step 7)")
    return chain


async def stream(llm: "LLM", chain: list[Model], tries: Callable[[Model], AsyncGenerator[dict | Final[A], None]]
                 ) -> AsyncGenerator[dict | Final[A], None]:
    """Each try's chunks, then the answering try's Final, stamped. Each try waits for the limiter first."""
    started = time.monotonic()
    candidate = chain[0]
    attempt = Attempt(candidate.provider, candidate.model, 0.0)
    for n in range(1, TRIES + 1):
        await llm.limiter.wait(candidate.provider, candidate.model)
        began, final = time.monotonic(), None  # an attempt's latency counts its tries, not the waits
        try:
            async with aclosing(tries(candidate)) as items:
                async for item in items:
                    if isinstance(item, Final):
                        final = item
                    else:
                        yield item
        except Exception as e:
            attempt.latency += time.monotonic() - began
            attempt.errors.append(e)
            if n == TRIES or not retryable(e):
                raise
            await asyncio.sleep(pause(e, n))
            continue
        assert final is not None, "a try yields its Final last"
        ended = time.monotonic()
        attempt.latency += ended - began
        final.value.latency = ended - started
        final.value.attempts = [attempt]
        final.value.cost = price(final.value.usage, candidate.provider, candidate.model)
        yield final
        return


async def run(llm: "LLM", chain: list[Model], once: Callable[[Model], Awaitable[A]]) -> A:
    """stream() for a call with no chunks."""
    async def tries(candidate: Model) -> AsyncGenerator[Final[A], None]:
        yield Final(await once(candidate))

    async with aclosing(stream(llm, chain, tries)) as items:
        async for item in items:
            if isinstance(item, Final):
                return item.value
    raise AssertionError("stream() ends with a Final or raises")


def retryable(e: Exception) -> bool:
    """Transient: a rate limit, a gateway or upstream failure, or a provider fault that isn't a 4xx."""
    if isinstance(e, (RateLimitError, GatewayTimeoutError, UpstreamProviderError)):
        return True
    # a 2xx is a fault inside a 200: an unreadable body, or anthropic's overloaded_error mid-stream
    return isinstance(e, ProviderError) and not isinstance(e, CutCall) and not 400 <= (e.status_code or 0) < 500


def pause(e: Exception, n: int) -> float:
    """Seconds before try n + 1: the provider's Retry-After, else full jitter."""
    after = _seconds(e.retry_after) if isinstance(e, RateLimitError) and e.retry_after else None
    return min(after, 60) if after is not None else random.uniform(0, 0.5 * 2 ** (n - 1))


def _seconds(retry_after: str) -> float | None:
    """Retry-After is a number of seconds or an HTTP date."""
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None
