"""async_query(): many query() calls at once, each result as soon as its request finishes. See SPEC.md Fan-out."""
import asyncio
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import TYPE_CHECKING, NotRequired, TypedDict

from llm_master import pipeline
from llm_master.io.messages import Message
from llm_master.io.response import Response
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM


class Request(TypedDict):
    """One query() call's input. Config and fallbacks live on the candidates."""

    messages: Sequence[Message | dict]
    moderation: NotRequired[bool]


async def run(llm: "LLM", candidates: Model | Sequence[Model], requests: Iterable[Request],
              concurrency: int | None) -> AsyncGenerator[tuple[int, Response | Exception], None]:
    requests, chain = list(requests), pipeline.chain(candidates)  # a batch's mistakes raise before anything is sent
    keys = Request.__required_keys__ | Request.__optional_keys__
    for request in requests:  # a dropped v1 key would change the answers
        if not Request.__required_keys__ <= request.keys() <= keys:
            raise ValueError(f"a request takes {sorted(Request.__required_keys__)}, and may take "
                             f"{sorted(Request.__optional_keys__)}, not {sorted(request)}: config and fallbacks "
                             "live on the candidates")
    if concurrency is not None and concurrency < 1:  # 0 would hold every request forever
        raise ValueError(f"concurrency is 1 or more, or None for 128, not {concurrency}")
    gate = asyncio.Semaphore(128 if concurrency is None else concurrency)

    async def one(index: int, request: Request) -> tuple[int, Response | Exception]:
        async with gate:
            try:
                return index, await llm.query(chain, request["messages"],
                                              moderation=request.get("moderation", False))
            except Exception as e:  # this request's result: the others go on
                return index, e

    tasks = [asyncio.create_task(one(index, request)) for index, request in enumerate(requests)]
    try:
        for finished in asyncio.as_completed(tasks):
            yield await finished
    finally:  # left early, or done: cancel what still runs, and wait for it
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
