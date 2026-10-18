"""max_tokens: one rename on responses, and anthropic's required default. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(n: int | None, api: API) -> dict:
    match api:
        case "completion":
            return {} if n is None else {"max_tokens": n}
        case "messages":
            return {"max_tokens": 64000 if n is None else n}  # required: the sonnet-line ceiling, so it never clips
        case "responses":
            return {} if n is None else {"max_output_tokens": n}
        case _:
            assert_never(api)
