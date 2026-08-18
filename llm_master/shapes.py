"""Row shapes for the providers page — pyright checks the page against these as you edit."""
from typing import NotRequired, TypedDict

from llm_master.providers import API, SearchStance


class ModelRow(TypedDict):
    """Rate-limit overrides; an absent key inherits the provider default."""

    requests: NotRequired[int]
    interval: NotRequired[int]  # seconds


class ProviderRow(TypedDict):
    api: API                       # which of the three APIs it speaks
    search: SearchStance           # its web-search stance
    requests: int
    interval: int                  # seconds
    models: dict[str, ModelRow]    # exceptions only; {} means none
    price_id: NotRequired[str]     # genai-prices name, only when it differs
