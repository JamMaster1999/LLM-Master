"""The providers. Pure data: every roster the repo derives its types from lives on this page, with the
row shape pyright checks the page against as you edit. Reads and import-time checks are in registry.py.
"""
from typing import Literal, NotRequired, TypedDict

API = Literal["completion", "messages", "responses"]
ProviderName = Literal["openai", "anthropic", "gemini", "vertexai", "fireworks"]
SDK = Literal["openai", "anthropic", "genai", "fireworks"]


class ProviderRow(TypedDict):
    api: API                       # which of the three APIs it speaks
    sdk: SDK                       # the SDK under its any-llm client; decides the aiohttp transport
    audio_api: NotRequired[API]    # where audio input goes, only when api takes none
    price_id: NotRequired[str]     # genai-prices name, only when it differs


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
