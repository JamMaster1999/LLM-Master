"""Row shapes for the providers page — pyright checks the page against these as you edit."""
from typing import NotRequired, TypedDict

from llm_master.providers import API, SDK


class ProviderRow(TypedDict):
    api: API                       # which of the three APIs it speaks
    sdk: SDK                       # the SDK under its any-llm client; decides the aiohttp transport
    audio_api: NotRequired[API]    # where audio input goes, only when api takes none
    price_id: NotRequired[str]     # genai-prices name, only when it differs
