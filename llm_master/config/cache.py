"""cache: auto / none / a lifetime in minutes. See CONFIG.md."""
from typing import Annotated, Literal, assert_never

from pydantic import Field

from llm_master.config import UnsupportedField
from llm_master.providers import API

Cache = Literal["auto", "none"] | Annotated[int, Field(ge=5, le=60)]  # int = minutes
_ANTHROPIC_TTLS = {5: "5m", 60: "1h"}


def translate(cache: Cache, api: API) -> dict:
    match api:
        case "completion":
            if isinstance(cache, int):
                raise UnsupportedField(api, "cache", "gemini and fireworks cache implicitly: there is no lifetime to set")
            return {}  # none is a quiet no-op: gemini's auto-cache has no off switch and no write fee
        case "messages":
            if isinstance(cache, int):  # anthropic judges any tier between its two
                return {"cache_control": {"type": "ephemeral", "ttl": _ANTHROPIC_TTLS.get(cache, f"{cache}m")}}
            return {}  # anthropic caches nothing unasked
        case "responses":  # extra_body: any-llm's ResponsesParams has no prompt_cache_options yet
            if isinstance(cache, int):  # assembly places the breakpoint explicit mode needs
                return {"extra_body": {"prompt_cache_options": {"mode": "explicit", "ttl": f"{cache}m"}}}
            if cache == "none":  # explicit mode with no breakpoint: no cache, no write fee
                return {"extra_body": {"prompt_cache_options": {"mode": "explicit"}}}
            return {}
        case _:
            assert_never(api)
