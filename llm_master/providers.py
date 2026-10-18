"""The providers. Pure data: every roster the repo derives its types from lives on this page.

Row shapes are in shapes.py; reads and import-time checks are in registry.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from llm_master.shapes import ModelRow, ProviderRow

API = Literal["completion", "messages", "responses"]
ProviderName = Literal["openai", "anthropic", "gemini", "vertexai", "fireworks"]
SearchStance = Literal["tool", "none"]
SDK = Literal["openai", "anthropic", "genai"]

_GEMINI_MODELS: dict[str, ModelRow] = {
    "gemini-2.5-flash-lite": {"requests": 30000},
    "gemini-2.5-flash-lite-preview-09-2025": {"requests": 30000},
    "gemini-3.1-flash-lite": {"requests": 30000},
    "gemini-3.1-flash-lite-preview": {"requests": 30000},
    "gemini-3.5-flash": {"requests": 20000},
    "gemini-3-flash-preview": {"requests": 20000},
    "gemini-2.5-pro": {"requests": 2000},
    "gemini-3-pro-preview": {"requests": 2000},
    "gemini-3.1-pro-preview": {"requests": 2000},
    "gemini-2.5-flash-image": {"requests": 5000},
    "gemini-3-pro-image": {"requests": 2000},
}

PROVIDERS: dict[ProviderName, ProviderRow] = {
    "openai": {
        "api": "responses",
        "search": "tool",
        "sdk": "openai",
        "requests": 10000,
        "interval": 60,
        "models": {
            "gpt-5.1": {"requests": 15000},
            "gpt-5.2": {"requests": 15000},
            "gpt-5.4": {"requests": 15000},
            "gpt-5.5": {"requests": 15000},
            "gpt-5.6-sol": {"requests": 15000},
            "gpt-5.6-terra": {"requests": 15000},
            "gpt-5.6-luna": {"requests": 15000},
            "chatgpt-4o-latest": {"requests": 5000},
            "gpt-audio-mini": {"requests": 5000},
            "gpt-4o-audio-preview": {"requests": 5000},
            "omni-moderation-latest": {"requests": 1000},
            "gpt-image-1": {"requests": 50},
        },
    },
    "anthropic": {
        "api": "messages",
        "search": "tool",
        "sdk": "anthropic",
        "requests": 4000,
        "interval": 60,
        "models": {},
    },
    "gemini": {
        "api": "completion",
        "search": "tool",
        "sdk": "genai",
        "price_id": "google",
        "requests": 10000,
        "interval": 60,
        "models": _GEMINI_MODELS,
    },
    "vertexai": {
        "api": "completion",
        "search": "tool",
        "sdk": "genai",
        "price_id": "google",
        "requests": 10000,
        "interval": 60,
        "models": _GEMINI_MODELS,
    },
    "fireworks": {
        "api": "completion",
        "search": "none",
        "sdk": "openai",
        "requests": 600,
        "interval": 60,
        "models": {},
    },
}
