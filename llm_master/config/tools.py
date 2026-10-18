"""tools: the definitions, reshaped per API. The call/result loop is io. See CONFIG.md."""
from typing import Annotated, Any, assert_never

from any_llm.tools import callable_to_tool
from pydantic import AfterValidator

from llm_master.providers import API


def _function_only(tool: Any) -> Any:
    if isinstance(tool, dict) and tool.get("type") != "function":
        raise ValueError(f"function tools only, got type={tool.get('type')!r}: provider-hosted tools are not supported")
    return tool


Tool = Annotated[Any, AfterValidator(_function_only)]  # an openai-format function dict or a python callable


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
