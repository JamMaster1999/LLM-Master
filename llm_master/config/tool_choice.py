"""tool_choice: who picks the tool. Card 9 in CONFIG.md."""
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from llm_master.config import API


class ToolChoice(BaseModel, extra="forbid"):
    """All fields optional, all mixable; unset = provider default.

    ToolChoice(mode="required")                      # must call some tool
    ToolChoice(only=["get_time"])                    # may use only these
    ToolChoice(mode="none")                          # no tools this turn
    ToolChoice(mode="required", only="get_time", parallel=False)  # exactly one call, exactly this tool
    """

    mode: Literal["auto", "none", "required"] | None = None
    only: str | list[str] | None = None   # restrict the menu to one name or a set
    parallel: bool | None = None          # False = at most one tool call per turn

    @field_validator("only")
    @classmethod
    def _singleton_is_its_name(cls, v):
        if isinstance(v, list) and not v:
            raise ValueError("only=[] allows nothing: use mode='none' to forbid tools")
        if isinstance(v, list) and len(v) == 1:
            return v[0]  # one meaning, one translation: ["f"] is "f"
        return v

    @model_validator(mode="after")
    def _none_forbids_only(self):
        if self.mode == "none" and self.only is not None:
            raise ValueError("mode='none' forbids only: no tools will run")
        return self


def translate(choice: ToolChoice, api: API) -> dict:
    if api == "messages":
        return _anthropic(choice)
    return _openai(choice, flat=api == "responses")


def _anthropic(c: ToolChoice) -> dict:
    if c.mode is None and c.only is None and c.parallel is None:
        return {}
    if c.mode == "required" and isinstance(c.only, str):
        tc = {"type": "tool", "name": c.only}
    else:  # anthropic has no subset form: a multi-name only collapses to forcing any tool (card 9)
        tc = {"type": "any" if c.mode == "required" else c.mode or "auto"}
    if c.parallel is not None:
        tc["disable_parallel_tool_use"] = not c.parallel
    return {"tool_choice": tc}


def _openai(c: ToolChoice, flat: bool) -> dict:
    out: dict = {}
    if c.parallel is not None:
        out["parallel_tool_calls"] = c.parallel
    if c.only is None:
        if c.mode is not None:
            out["tool_choice"] = c.mode
    elif c.mode == "required" and isinstance(c.only, str):
        out["tool_choice"] = _tool(c.only, flat)
    else:
        names = [c.only] if isinstance(c.only, str) else c.only
        allowed = {"mode": c.mode or "auto", "tools": [_tool(n, flat) for n in names]}
        out["tool_choice"] = (
            {"type": "allowed_tools", **allowed} if flat
            else {"type": "allowed_tools", "allowed_tools": allowed}
        )
    return out


def _tool(name: str, flat: bool) -> dict:
    """responses spells a tool flat; chat completions nests it."""
    return {"type": "function", "name": name} if flat else {"type": "function", "function": {"name": name}}
