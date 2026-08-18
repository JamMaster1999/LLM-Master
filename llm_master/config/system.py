"""system: the system prompt. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(text: str, api: API) -> dict:
    match api:
        case "completion":
            return {}  # no param exists: prepended as a system-role message at assembly (io/messages)
        case "messages":
            return {"system": text}
        case "responses":
            return {"instructions": text}
        case _:
            assert_never(api)
