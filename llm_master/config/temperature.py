"""temperature: same name on every API. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(t: float, api: API) -> dict:
    match api:
        case "completion":
            return {"temperature": t}
        case "messages":
            return {"temperature": t}
        case "responses":
            return {"temperature": t}
        case _:
            assert_never(api)
