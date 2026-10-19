"""system: the system prompt. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(text: str | None, api: API) -> dict:
    match api:
        case "completion":
            return {}  # no param exists: prepended as a system-role message at assembly (io/messages)
        case "messages":
            return {} if text is None else {"system": text}
        case "responses":
            return {} if text is None else {"instructions": text}
        case _:
            assert_never(api)
