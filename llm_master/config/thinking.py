"""thinking: one word in, each API's reasoning params out. See CONFIG.md."""
from typing import Literal, assert_never

from llm_master.providers import API

Level = Literal["none", "low", "medium", "high", "xhigh"]


def translate(level: Level, api: API) -> dict:
    match api:
        case "completion":
            return {"reasoning_effort": level}
        case "messages":
            if level == "none":
                return {"thinking": {"type": "disabled"}}
            # Claude 5.x hides the thought text unless asked: display defaults to omitted
            return {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": level}}
        case "responses":
            if level == "none":
                return {"reasoning": {"effort": "none"}}
            # summary always fetched: openai's thought text is opt-in
            return {"reasoning": {"effort": level, "summary": "auto"}}
        case _:
            assert_never(api)
