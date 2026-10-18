"""timeout: seconds, forwarded to each provider SDK. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(seconds: float | None, api: API) -> dict:
    match api:
        case "completion":
            return {} if seconds is None else {"timeout": seconds}
        case "messages":
            # unset: anthropic's own 10-minute request ceiling, said aloud, because the SDK's
            # long-request guard rejects a non-streaming 64k max_tokens unless a timeout is passed
            return {"timeout": 600 if seconds is None else seconds}
        case "responses":
            return {} if seconds is None else {"timeout": seconds}
        case _:
            assert_never(api)
