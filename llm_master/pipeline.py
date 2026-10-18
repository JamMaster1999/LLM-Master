"""The pipeline every call runs through: query(), image() and speech() hand it one try per candidate.

A try is an async generator: a streamed try yields its chunks (dicts), and every try yields its result last.
Steps 5 to 7 grow this loop in place: the rate limiter, retries, then fallback over the candidates.
"""
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from llm_master.io.response import Attempt
from llm_master.types import Model

T = TypeVar("T")


@dataclass
class Done(Generic[T]):
    """The answering try's result, every attempt (the last one answered), and the whole call's latency."""

    result: T
    attempts: list[Attempt]
    latency: float


def chain(candidates: Model | Sequence[Model]) -> list[Model]:
    chain = [candidates] if isinstance(candidates, Model) else list(candidates)
    if len(chain) != 1:  # never drop a fallback silently: it lands in SPEC.md step 7
        raise NotImplementedError(f"{len(chain)} candidates: a call takes exactly one until fallback (step 7)")
    return chain


async def stream(chain: list[Model], tries: Callable[[Model], AsyncGenerator[dict | T, None]]
                 ) -> AsyncGenerator[dict | Done[T], None]:
    """The try's chunks, a reset-step when a streamed try fails, then the Done."""
    started = time.monotonic()
    candidate, opened, result = chain[0], False, None
    try:
        async with aclosing(tries(candidate)) as items:
            async for item in items:
                if isinstance(item, dict):
                    opened = True
                    yield item
                else:
                    result = item
    except Exception:
        if opened:
            yield {"type": "reset-step"}  # the browser drops this try's output
        raise
    latency = time.monotonic() - started
    yield Done(cast(T, result), [Attempt(candidate.provider, candidate.model, latency)], latency)


async def run(chain: list[Model], once: Callable[[Model], Awaitable[T]]) -> Done[T]:
    """stream() for a call with no chunks."""
    async def tries(candidate: Model) -> AsyncGenerator[T, None]:
        yield await once(candidate)

    async with aclosing(stream(chain, tries)) as items:
        async for item in items:
            if isinstance(item, Done):
                return item
    raise AssertionError("stream() ends with a Done or raises")
