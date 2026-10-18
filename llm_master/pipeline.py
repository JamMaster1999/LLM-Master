"""The pipeline every call runs through: query(), image() and speech() hand it one try per candidate.

A try is an async generator: a streamed try yields its chunks, and every try yields its Result last. The
pipeline stamps every attempt and the whole call's latency on that result. Steps 5 to 7 grow this loop in
place: the rate limiter, retries, then fallback over the candidates.
"""
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from llm_master.io.response import Attempt
from llm_master.types import Model


class Answer(Protocol):
    """What a call returns, a Response or a Media: both carry these."""

    attempts: list[Attempt]
    latency: float


A = TypeVar("A", bound=Answer)


@dataclass
class Result(Generic[A]):
    """What a try yields last. The wrapper keeps a result from ever passing for a chunk."""

    value: A


def chain(candidates: Model | Sequence[Model]) -> list[Model]:
    chain = [candidates] if isinstance(candidates, Model) else list(candidates)
    if len(chain) != 1:  # never drop a fallback silently: it lands in SPEC.md step 7
        raise NotImplementedError(f"{len(chain)} candidates: a call takes exactly one until fallback (step 7)")
    return chain


async def stream(chain: list[Model], tries: Callable[[Model], AsyncGenerator[dict | Result[A], None]]
                 ) -> AsyncGenerator[dict | Result[A], None]:
    """The try's chunks, then its Result, stamped."""
    started = time.monotonic()
    candidate, result = chain[0], None
    async with aclosing(tries(candidate)) as items:
        async for item in items:
            if isinstance(item, Result):
                result = item
            else:
                yield item
    assert result is not None, "a try yields its Result last"
    result.value.latency = time.monotonic() - started
    result.value.attempts = [Attempt(candidate.provider, candidate.model, result.value.latency)]
    yield result


async def run(chain: list[Model], once: Callable[[Model], Awaitable[A]]) -> A:
    """stream() for a call with no chunks."""
    async def tries(candidate: Model) -> AsyncGenerator[Result[A], None]:
        yield Result(await once(candidate))

    async with aclosing(stream(chain, tries)) as items:
        async for item in items:
            if isinstance(item, Result):
                return item.value
    raise AssertionError("stream() ends with a Result or raises")
