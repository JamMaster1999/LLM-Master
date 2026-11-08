"""The pipeline every call runs through: query(), image() and speech() hand it one try per candidate.

A try is an async generator: a streamed try yields its chunks, and every try yields its Final last. The
pipeline stamps every attempt, the whole call's latency and its cost on that CallReturn. Per try, the rate
limiter first, then the try, then retries (SPEC.md Failures). Step 7 adds fallback over the candidates.
"""
import asyncio
import random
import time
import traceback
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from any_llm.exceptions import GatewayTimeoutError, ProviderError, RateLimitError, UpstreamProviderError

from llm_master.cost import price
from llm_master.io.response import Attempt, CutCall, Usage
from llm_master.providers import PROVIDERS
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
T = TypeVar("T")
TRIES = 5  # per candidate
BUDGET = 120.0  # the longest a candidate's waits between tries add up to
QUIET = 600.0  # a genai try with no timeout set fails after this long without an item


@dataclass
class Final(Generic[A]):
    """A try's last item, its CallReturn. The label keeps it from ever passing for a chunk."""

    value: A


def chain(candidates: Model | Sequence[Model]) -> list[Model]:
    chain = [candidates] if isinstance(candidates, Model) else list(candidates)
    if len(chain) != 1:  # never drop a fallback silently: it lands in SPEC.md step 7
        raise NotImplementedError(f"{len(chain)} candidates: a call takes exactly one until fallback (step 7)")
    return chain


async def stream(llm: "LLM", chain: list[Model], once: Callable[[Model], AsyncGenerator[dict | Final[A], None]]
                 ) -> AsyncGenerator[dict | Final[A], None]:
    """Each try's chunks, then the answering try's Final, stamped. Each try waits for the limiter first."""
    started = time.monotonic()
    candidate = chain[0]
    attempt = Attempt(candidate.provider, candidate.model, 0.0)
    waited, quiet = 0.0, _quiet(candidate)  # waited counts only the pauses: not the tries, not the limiter
    for n in range(1, TRIES + 1):
        await llm.limiter.wait(candidate.provider, candidate.model)
        began, final = time.monotonic(), None  # an attempt's latency counts its tries, not the waits
        try:
            async with aclosing(once(candidate)) as items:
                while (item := await _next(items, quiet, candidate.provider)) is not None:
                    if isinstance(item, Final):
                        final = item
                    else:
                        yield item
        except Exception as e:
            attempt.latency += time.monotonic() - began
            _unload(e)  # the error stays on the attempt; its request needn't
            attempt.errors.append(e)
            if (pause := _pause(e, n)) is None or waited + pause > BUDGET:
                raise
            waited += pause
            await asyncio.sleep(pause)
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
    async def streamed(candidate: Model) -> AsyncGenerator[Final[A], None]:
        yield Final(await once(candidate))

    async with aclosing(stream(llm, chain, streamed)) as items:
        async for item in items:
            if isinstance(item, Final):
                return item.value
    raise AssertionError("stream() ends with a Final or raises")


def _quiet(candidate: Model) -> float | None:
    """genai sets no timeout of its own, and its timeout would cap a whole stream, so a genai candidate with none
    set gets QUIET seconds for each item instead: its answer, or a stream's next chunk."""
    unset = candidate.config.timeout is None
    return QUIET if unset and PROVIDERS[candidate.provider]["sdk"] == "genai" else None


async def _next(items: AsyncGenerator[T, None], quiet: float | None, provider: str) -> T | None:
    """items' next item, None at its end, or a retryable ProviderError when it takes longer than quiet seconds."""
    try:
        async with asyncio.timeout(quiet):
            return await anext(items, None)
    except TimeoutError as e:
        raise ProviderError(f"nothing came in {quiet:g} seconds", provider_name=provider) from e


def _retryable(e: Exception) -> bool:
    """Transient: a rate limit, a gateway or upstream failure, or a provider fault that isn't a 4xx."""
    if isinstance(e, (RateLimitError, GatewayTimeoutError, UpstreamProviderError)):
        return True
    # a 2xx is a fault inside a 200: an unreadable body, or anthropic's overloaded_error mid-stream
    return isinstance(e, ProviderError) and not isinstance(e, CutCall) and not 400 <= (e.status_code or 0) < 500


def _pause(e: Exception, n: int) -> float | None:
    """Seconds before try n + 1, after try n failed with e, or None when the candidate stops: its last try, or an
    error a retry would repeat. A 429 waits as long as the provider asks; anything else, full jitter."""
    if n == TRIES or not _retryable(e):
        return None
    if isinstance(e, RateLimitError) and (after := _after(e)) is not None:
        return after
    return random.uniform(0, 0.5 * 2 ** (n - 1))


def _after(e: RateLimitError) -> float | None:
    """The provider's Retry-After in seconds, a number or an HTTP date; None when it sent none."""
    if not (after := e.retry_after or _retry_delay(e.original_exception)):
        return None
    try:
        return max(0.0, float(after))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(after) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def _retry_delay(e: Exception | None) -> str | None:
    """Gemini's Retry-After, which it sends in the 429's body as a RetryInfo's retryDelay, such as "20s". Remove
    once any-llm reads it into retry_after: https://github.com/mozilla-ai/any-llm/pull/1622"""
    body = getattr(e, "details", None)  # google-genai's error keeps its JSON body there
    infos = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    return next((i["retryDelay"].removesuffix("s") for i in infos
                 if i.get("@type") == "type.googleapis.com/google.rpc.RetryInfo" and "retryDelay" in i), None)


def _unload(e: BaseException) -> None:
    """Drops what a kept error holds of its request: its frames' locals, and the HTTP exchange each error in its
    chain keeps, the SDK's included, with the request body and its media. Request ids, statuses and codes stay."""
    seen, todo = set(), [e]
    while todo:
        x = todo.pop()
        if x is None or id(x) in seen:
            continue
        seen.add(id(x))
        traceback.clear_frames(x.__traceback__)
        for name in ("request", "_request", "response", "_response"):  # where SDK and httpx errors keep it
            if name in vars(x):
                setattr(x, name, None)
        todo += [getattr(x, "original_exception", None), x.__cause__, x.__context__]
