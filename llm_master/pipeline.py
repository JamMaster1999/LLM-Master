"""The pipeline every call runs through: query(), image() and speech() hand it one try per candidate.

Steps 5 to 7 grow it in place: the rate limiter, retries, then fallback over the candidates.
"""
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from llm_master.io.response import Attempt
from llm_master.types import Model

T = TypeVar("T")


def chain(candidates: Model | Sequence[Model]) -> list[Model]:
    chain = [candidates] if isinstance(candidates, Model) else list(candidates)
    if len(chain) != 1:  # never drop a fallback silently: it lands in SPEC.md step 7
        raise NotImplementedError(f"{len(chain)} candidates: fallback chains arrive in step 7")
    return chain


async def run(candidates: Model | Sequence[Model], once: Callable[[Model], Awaitable[T]]
              ) -> tuple[T, Model, list[Attempt], float]:
    """once(candidate)'s result, the candidate that gave it, every attempt, and the whole call's latency."""
    started = time.monotonic()
    candidate = chain(candidates)[0]
    result = await once(candidate)
    latency = time.monotonic() - started
    return result, candidate, [Attempt(candidate.provider, candidate.model, latency)], latency
