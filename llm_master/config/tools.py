"""tools: the definitions, reshaped per API. The call/result loop is io. Card 8 in CONFIG.md."""
from typing import Any

from any_llm.tools import callable_to_tool

from llm_master.config import API


def translate(tools: list[Any], api: API) -> dict:
    specs = [tool if isinstance(tool, dict) else callable_to_tool(tool) for tool in tools]
    if api == "completion":
        return {"tools": specs}  # openai nested form as-is; any-llm converts per provider
    functions = [dict(spec["function"]) for spec in specs]
    if api == "messages":
        for f in functions:  # rename only; every other key rides through and the provider judges it
            f["input_schema"] = f.pop("parameters", {})
        return {"tools": functions}
    return {"tools": [{"type": "function", **f} for f in functions]}  # responses takes tools flat
