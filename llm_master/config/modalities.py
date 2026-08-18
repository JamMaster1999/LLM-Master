"""modalities: what the reply may contain — non-text arms per API below. See CONFIG.md."""
from typing import Literal, assert_never

from llm_master.config import UnsupportedField
from llm_master.providers import API

Modality = Literal["text", "image", "audio"]


def translate(modalities: list[Modality], api: API) -> dict:
    extras = sorted(set(modalities) - {"text"})
    match api:
        case "completion":
            if not extras:
                return {}  # text only: every provider's default, nothing to send
            raise UnsupportedField(api, "modalities",
                                   f"no completion spelling for {extras} yet — per-provider gates in CONFIG.md")
        case "messages":
            if not extras:
                return {}
            raise UnsupportedField(api, "modalities", "anthropic outputs text only — permanent")
        case "responses":
            if not extras:
                return {}
            raise UnsupportedField(api, "modalities",
                                   f"{extras} unverified on openai — arm opens after a live seed")
        case _:
            assert_never(api)
