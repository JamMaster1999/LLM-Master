"""The providers. Pure data: every roster the repo derives its types from lives on this page.

Row shapes are in shapes.py; reads and import-time checks are in registry.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from llm_master.shapes import ProviderRow

API = Literal["completion", "messages", "responses"]
ProviderName = Literal["openai", "anthropic", "gemini", "vertexai", "fireworks"]
SDK = Literal["openai", "anthropic", "genai", "fireworks"]

PROVIDERS: dict[ProviderName, ProviderRow] = {
    "openai": {
        "api": "responses",
        "audio_api": "completion",
        "sdk": "openai",
    },
    "anthropic": {
        "api": "messages",
        "sdk": "anthropic",
    },
    "gemini": {
        "api": "completion",
        "sdk": "genai",
        "price_id": "google",
    },
    "vertexai": {
        "api": "completion",
        "sdk": "genai",
        "price_id": "google",
    },
    "fireworks": {
        "api": "completion",
        "sdk": "fireworks",
    },
}
