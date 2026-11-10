"""async_query(): many query() calls at once, each result as soon as its request finishes. See SPEC.md Fan-out."""
import asyncio
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import TYPE_CHECKING

from llm_master import pipeline
from llm_master.io.messages import Message
from llm_master.io.response import Response
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM


async def run(llm: "LLM", candidates: Model | Sequence[Model], requests: Iterable[Sequence[Message | dict]],
              concurrency: int, moderation: bool) -> AsyncGenerator[tuple[int, Response | Exception], None]:
    chain = pipeline.chain(candidates)  # a batch's mistakes raise before anything is sent
    if concurrency < 1:  # 0 would hold every request forever
        raise ValueError(f"concurrency is 1 or more, not {concurrency}")
    gate = asyncio.Semaphore(concurrency)

    async def one(index: int, messages: Sequence[Message | dict]) -> tuple[int, Response | Exception]:
        async with gate:
            try:
                return index, await llm.query(chain, messages, moderation=moderation)
            except Exception as e:  # this request's result: the others go on
                return index, e

    tasks = [asyncio.create_task(one(index, messages)) for index, messages in enumerate(requests)]
    try:
        for finished in asyncio.as_completed(tasks):
            yield await finished
    finally:  # left early, or done: cancel what still runs, and wait for it
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
