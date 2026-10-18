"""What you write per call: Model, one fallback candidate, and Config, its tunables."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_master.config.response_format import Format
from llm_master.config.thinking import Level
from llm_master.config.tool_choice import ToolChoice
from llm_master.config.tools import Tool
from llm_master.providers import ProviderName


class Model(BaseModel, extra="forbid"):
    """One fallback candidate: where it runs and the config it runs with."""

    provider: ProviderName
    model: str
    config: Config = Field(default_factory=lambda: Config())  # omitted = all defaults
    extra: dict[str, Any] = {}  # raw pass-through, merged last


class Config(BaseModel, extra="forbid"):
    """The tunable fields, in authoring order."""

    system: str | None = None
    temperature: float | None = None
    thinking: Level | None = None
    tools: list[Tool] | None = None  # openai-format function dicts or python callables
    tool_choice: ToolChoice | None = None
    response_format: Format | None = None  # type="text" | "json" | "schema" (with schema=...)
    max_tokens: int | None = Field(None, gt=0)
    timeout: float | None = Field(None, gt=0)
    cache: Literal["auto", "none"] | int = "auto"  # int = minutes, 5-60
    moderation: bool = False
