"""stream: deliver as a stream. See CONFIG.md."""
from typing import assert_never

from llm_master.providers import API


def translate(on: bool, api: API) -> dict:
    match api:
        case "completion":
            # fireworks ends a stream with no usage at all without include_usage (live-verified)
            return {"stream": True, "stream_options": {"include_usage": True}} if on else {}
        case "messages":
            return {"stream": True} if on else {}
        case "responses":
            return {"stream": True} if on else {}
        case _:
            assert_never(api)
