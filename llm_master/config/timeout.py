"""timeout: seconds, forwarded to each provider SDK. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(seconds: float, api: API) -> dict:
    match api:
        case "completion":
            return {"timeout": seconds}
        case "messages":
            return {"timeout": seconds}
        case "responses":
            return {"timeout": seconds}
        case _:
            assert_never(api)
