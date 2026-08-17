"""modalities: what the reply may contain. Card 4 in CONFIG.md — non-text arms per API below."""
from typing import Literal, assert_never

from llm_master.config import API, UnsupportedField

Modality = Literal["text", "image", "audio"]


def translate(modalities: list[Modality], api: API) -> dict:
    extras = sorted(set(modalities) - {"text"})
    if not extras:
        return {}
    match api:
        case "completion":
            raise UnsupportedField(api, "modalities",
                                   f"{extras} gated: gemini media drops in any-llm (#1295), "
                                   "fireworks lacks the capability")
        case "messages":
            raise UnsupportedField(api, "modalities", "anthropic outputs text only — permanent")
        case "responses":
            raise UnsupportedField(api, "modalities",
                                   f"{extras} unverified on openai — arm opens after a live seed")
        case _:
            assert_never(api)
