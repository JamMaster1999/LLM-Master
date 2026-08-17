"""The assemblies: provider registry rows, then what you write per call. Field types live in config/."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from llm_master.config import API
from llm_master.config.modalities import Modality
from llm_master.config.response_format import Format
from llm_master.config.thinking import Level
from llm_master.config.tool_choice import ToolChoice
from llm_master.config.web_search import Search, SearchStance


class Provider(BaseModel, frozen=True, extra="forbid"):
    name: str
    api: API                                   # required: which of the three APIs this provider speaks
    search: SearchStance                       # required: every provider takes a search decision
    requests: int = Field(gt=0)
    interval: int = Field(gt=0)                # seconds
    models: tuple[ModelLimit, ...]             # exceptions only; () means none
    price_id: str | None = None                # genai-prices name, only when it differs

    @model_validator(mode="after")
    def _no_duplicate_models(self):
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate model entries")
        return self


class ModelLimit(BaseModel, frozen=True, extra="forbid"):
    name: str
    requests: int | None = Field(None, gt=0)   # None = inherit from provider
    interval: int | None = Field(None, gt=0)   # seconds

    @model_validator(mode="after")
    def _overrides_something(self):
        if self.requests is None and self.interval is None:
            raise ValueError(f"{self.name}: overrides nothing")
        return self


class Model(BaseModel, extra="forbid"):
    """One fallback candidate: where it runs and the config it runs with."""

    provider: Literal["openai", "anthropic", "gemini", "vertexai", "fireworks"]
    model: str
    config: Config = Field(default_factory=lambda: Config())  # omitted = all defaults
    extra: dict[str, Any] = {}  # raw pass-through, merged last


class Config(BaseModel, extra="forbid"):
    """The tunable fields, in authoring order."""

    system: str | None = None
    modalities: list[Modality] | None = None
    temperature: float | None = None
    thinking: Level | None = None
    thinking_summaries: bool = False
    tools: list[Any] | None = None  # openai-format dicts or python callables
    tool_choice: ToolChoice | None = None
    web_search: Search | None = None
    response_format: Format | None = None  # "json", a pydantic model class, or a json-schema dict
    max_tokens: int | None = Field(None, gt=0)
    timeout: float | None = Field(None, gt=0)
    cache: Literal["auto", "none"] | int = "auto"  # int = minutes, 5-60
    moderation: bool = False
