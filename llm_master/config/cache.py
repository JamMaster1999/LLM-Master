"""cache: on (the default) or off. Off skips the cache write fee on one-shot prompts. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(on: bool, api: API) -> dict:
    match api:
        case "completion":
            return {}  # gemini and fireworks cache on their own and charge no write fee
        case "messages":
            return {"cache_control": {"type": "ephemeral"}} if on else {}  # anthropic caches only when asked
        case "responses":  # explicit mode with no breakpoint caches nothing, so it writes nothing
            return {} if on else {"extra_body": {"prompt_cache_options": {"mode": "explicit"}}}
        case _:
            assert_never(api)
