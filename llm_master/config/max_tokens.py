"""max_tokens: one rename on responses. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(n: int, api: API) -> dict:
    match api:
        case "completion":
            return {"max_tokens": n}
        case "messages":
            return {"max_tokens": n}
        case "responses":
            return {"max_output_tokens": n}
        case _:
            assert_never(api)
