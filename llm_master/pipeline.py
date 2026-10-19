"""The pipeline every call runs through: query(), image() and speech() hand it one try per candidate.

A try is an async generator: a streamed try yields its chunks, and every try yields its Final last. The
pipeline stamps every attempt, the whole call's latency and its cost on that CallReturn. Steps 5 to 7 grow this
loop in place: the rate limiter, retries, then fallback over the candidates.
"""
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

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
    """The try's chunks, then its Final, stamped. The try waits for the limiter first."""
    started = time.monotonic()
    candidate, final = chain[0], None
    await llm.limiter.wait(candidate.provider, candidate.model)
    began = time.monotonic()  # an attempt's latency starts with its try, after the limiter
    async with aclosing(tries(candidate)) as items:
        async for item in items:
            if isinstance(item, Final):
                final = item
            else:
                yield item
    assert final is not None, "a try yields its Final last"
    ended = time.monotonic()
    final.value.latency = ended - started
    final.value.attempts = [Attempt(candidate.provider, candidate.model, ended - began)]
    final.value.cost = price(final.value.usage, candidate.provider, candidate.model)
    yield final


async def run(llm: "LLM", chain: list[Model], once: Callable[[Model], Awaitable[A]]) -> A:
    """stream() for a call with no chunks."""
    async def tries(candidate: Model) -> AsyncGenerator[Final[A], None]:
        yield Final(await once(candidate))

    async with aclosing(stream(llm, chain, tries)) as items:
        async for item in items:
            if isinstance(item, Final):
                return item.value
    raise AssertionError("stream() ends with a Final or raises")
