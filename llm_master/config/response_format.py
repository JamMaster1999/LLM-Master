"""response_format: what shape the output must be. Card 11 in CONFIG.md.

None = text. "json" = valid JSON, any shape. A pydantic class or json-schema dict = that schema.
"""
from typing import Any, Literal

from llm_master.config import API, UnsupportedField

Format = Literal["json"] | type | dict[str, Any]


def translate(schema: Format, api: API) -> dict:
    if schema == "json":
        if api == "messages":
            raise UnsupportedField(api, "response_format",
                                   "anthropic has no schema-less JSON mode — pass a schema instead")
        return {"response_format": {"type": "json_object"}}
    return {"output_format": schema} if api == "messages" else {"response_format": schema}
