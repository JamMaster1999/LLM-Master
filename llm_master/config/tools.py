"""tools: the definitions, reshaped per API. The call/result loop is io. See CONFIG.md."""
from typing import Any, assert_never

from any_llm.tools import callable_to_tool

from llm_master.providers import API


def translate(tools: list[Any], api: API) -> dict:
    specs = [tool if isinstance(tool, dict) else callable_to_tool(tool) for tool in tools]
    match api:
        case "completion":
            return {"tools": specs}  # openai nested form as-is; any-llm converts per provider
        case "messages":
            functions = [dict(spec["function"]) for spec in specs]
            for f in functions:  # rename only; every other key rides through and the provider judges it
                f["input_schema"] = f.pop("parameters", {})
            return {"tools": functions}
        case "responses":
            return {"tools": [{"type": "function", **spec["function"]} for spec in specs]}  # flat form
        case _:
            assert_never(api)
