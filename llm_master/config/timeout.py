"""timeout: seconds, forwarded to each provider SDK, and genai's quiet timer. See CONFIG.md."""
import asyncio
from collections.abc import Awaitable
from typing import TypeVar, assert_never

from any_llm.exceptions import ProviderError

from llm_master.providers import API, PROVIDERS, ProviderName

T = TypeVar("T")
QUIET = 600.0  # the longest a genai read may take when its request sets no timeout


def translate(seconds: float | None, api: API) -> dict:
    match api:
        case "completion":
            return {} if seconds is None else {"timeout": seconds}
        case "messages":
            # unset: anthropic's own 10-minute request ceiling, said aloud, because the SDK's
            # long-request guard rejects a non-streaming 64k max_tokens unless a timeout is passed
            return {"timeout": 600 if seconds is None else seconds}
        case "responses":
            return {} if seconds is None else {"timeout": seconds}
        case _:
            assert_never(api)


async def bounded(read: Awaitable[T], provider: ProviderName, request: dict) -> T:
    """read's result. genai sets no timeout of its own, and its timeout would cap a whole stream, so a genai read
    whose request sets none fails as a retryable ProviderError after QUIET seconds: an answer, or a stream's next
    chunk."""
    if PROVIDERS[provider]["sdk"] != "genai" or "timeout" in request:
        return await read
    try:
        async with asyncio.timeout(QUIET):
            return await read
    except TimeoutError as e:
        raise ProviderError(f"nothing came in {QUIET:g} seconds", provider_name=provider) from e
