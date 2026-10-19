"""temperature: same name on every API. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(t: float | None, api: API) -> dict:
    match api:
        case "completion":
            return {} if t is None else {"temperature": t}
        case "messages":
            return {} if t is None else {"temperature": t}
        case "responses":
            return {} if t is None else {"temperature": t}
        case _:
            assert_never(api)
