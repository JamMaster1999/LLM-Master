"""async_query(): many query() calls at once, v1's signature over v2's candidates. See SPEC.md Fan-out."""
import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NotRequired, TypedDict

from llm_master.io.messages import Message, validate
from llm_master.io.response import Response
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM


class Request(TypedDict):
    """One query() call's input. Config and fallbacks live on the candidates."""

    messages: Sequence[Message | dict]
    moderation: NotRequired[bool]


async def run(llm: "LLM", candidates: Model | Sequence[Model], requests: Sequence[Request], concurrency: int | None,
              progress: Callable[[int, int], None] | None) -> list[Response]:
    keys = Request.__required_keys__ | Request.__optional_keys__
    for request in requests:  # before anything is sent: a dropped v1 key would change the answers
        if unknown := request.keys() - keys:
            raise ValueError(f"a request takes only {sorted(keys)}, not {sorted(unknown)}: config and fallbacks "
                             "live on the candidates")
    histories = [validate(request["messages"]) for request in requests]
    gate, done = asyncio.Semaphore(concurrency or 128), 0  # v1's default, and its reading of 0

    async def one(history: list[Message], moderation: bool) -> Response:
        nonlocal done
        async with gate:
            response = await llm.query(candidates, history, moderation=moderation)
        done += 1
        if progress:
            progress(done, len(requests))
        return response

    try:
        async with asyncio.TaskGroup() as group:  # the first failure cancels the rest
            tasks = [group.create_task(one(history, request.get("moderation", False)))
                     for history, request in zip(histories, requests)]
    except ExceptionGroup as failed:
        raise failed.exceptions[0] from None  # typed as a single call's error, not wrapped
    return [task.result() for task in tasks]
