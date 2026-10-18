"""max_tokens: one rename on responses, and anthropic's required default. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(n: int | None, api: API) -> dict:
    match api:
        case "completion":
            return {} if n is None else {"max_tokens": n}
        case "messages":
            # anthropic requires a number: the sonnet-line ceiling, plus its 10-minute request ceiling,
            # which any-llm's long-request guard demands for 64k. A caller's timeout merges later and wins.
            return {"max_tokens": 64000, "timeout": 600} if n is None else {"max_tokens": n}
        case "responses":
            return {} if n is None else {"max_output_tokens": n}
        case _:
            assert_never(api)
