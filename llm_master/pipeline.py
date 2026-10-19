"""The pipeline every call runs through: query(), image() and speech() hand it one try per candidate.

A try is an async generator: a streamed try yields its chunks, and every try yields its Final last. The
pipeline stamps every attempt, the whole call's latency and its cost on that CallReturn. Candidates run in
order; per try, the rate limiter first, then the try, then retries or the next candidate (SPEC.md Failures).
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
from llm_master.io.response import Attempt, Usage
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
BUDGET = 120.0  # the longest a candidate's waits between tries add up to


@dataclass
class Final(Generic[A]):
    """A try's last item, its CallReturn. The label keeps it from ever passing for a chunk."""

    value: A


def chain(candidates: Model | Sequence[Model]) -> list[Model]:
    chain = [candidates] if isinstance(candidates, Model) else list(candidates)
    if not chain:
        raise ValueError("a call needs at least one candidate")
    return chain


async def stream(llm: "LLM", chain: list[Model], once: Callable[[Model], AsyncGenerator[dict | Final[A], None]]
                 ) -> AsyncGenerator[dict | Final[A], None]:
    """Each try's chunks, then the answering try's Final, stamped. Each try waits for the limiter first."""
    started, attempts = time.monotonic(), []
    for candidate in chain:
        attempts.append(attempt := Attempt(candidate.provider, candidate.model, 0.0))
        waited = 0.0  # the pauses alone: not the tries, not the limiter
        for n in range(1, TRIES + 1):
            await llm.limiter.wait(candidate.provider, candidate.model)
            began, final = time.monotonic(), None  # an attempt's latency counts its tries, not the waits
            try:
                async with aclosing(once(candidate)) as items:
                    async for item in items:
                        if isinstance(item, Final):
                            final = item
                        else:
                            yield item
            except Exception as e:
                attempt.latency += time.monotonic() - began
                _unload(e)  # the error stays on the attempt; its request needn't
                attempt.errors.append(e)
                if (pause := _pause(e, n, waited)) is None:
                    break  # to the next candidate
                waited += pause
                await asyncio.sleep(pause)
                continue
            assert final is not None, "a try yields its Final last"
            ended = time.monotonic()
            attempt.latency += ended - began
            final.value.latency = ended - started
            final.value.attempts = attempts
            final.value.cost = price(final.value.usage, candidate.provider, candidate.model)
            yield final
            return
    raise _last(attempts)


async def run(llm: "LLM", chain: list[Model], once: Callable[[Model], Awaitable[A]]) -> A:
    """stream() for a call with no chunks."""
    async def streamed(candidate: Model) -> AsyncGenerator[Final[A], None]:
        yield Final(await once(candidate))

    async with aclosing(stream(llm, chain, streamed)) as items:
        async for item in items:
            if isinstance(item, Final):
                return item.value
    raise AssertionError("stream() ends with a Final or raises")


def _last(attempts: list[Attempt]) -> Exception:
    """The last candidate's last error, noting every earlier one: what a call raises when all fail."""
    errors = [(a, e) for a in attempts for e in a.errors]
    last = errors[-1][1]
    for a, e in errors[:-1]:
        last.add_note(f"earlier, {a.provider}/{a.model}: {type(e).__name__}: {e}")
    return last


def _retryable(e: Exception) -> bool:
    """Transient: a rate limit, a gateway or upstream failure, or a provider fault that isn't a 4xx."""
    if isinstance(e, (RateLimitError, GatewayTimeoutError, UpstreamProviderError)):
        return True
    # a 2xx is a fault inside a 200: an unreadable body, or anthropic's overloaded_error mid-stream
    return isinstance(e, ProviderError) and not 400 <= (e.status_code or 0) < 500


def _pause(e: Exception, n: int, waited: float) -> float | None:
    """Seconds before try n + 1, after try n failed with e and waited seconds of pauses, or None when the candidate
    stops: its last try, an error a retry would repeat, or a pause past the budget. A 429 waits as long as the
    provider asks; anything else, full jitter."""
    if n == TRIES or not _retryable(e):
        return None
    after = _after(e) if isinstance(e, RateLimitError) else None
    pause = random.uniform(0, 0.5 * 2 ** (n - 1)) if after is None else after
    return pause if waited + pause <= BUDGET else None


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
