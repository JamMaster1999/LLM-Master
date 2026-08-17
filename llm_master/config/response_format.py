"""response_format: what shape the output must be. Card 11 in CONFIG.md.

Flat like ToolChoice: type says the mode, schema carries the shape.
Format(type="json") = valid JSON, any shape. Format(type="schema", schema=X) = that shape
exactly, as a pydantic class or a json-schema dict. Format(type="text") = plain text — every
provider's default, so it emits nothing. None = unset.
"""
from typing import Any, Literal, assert_never

from pydantic import BaseModel, Field, model_validator

from llm_master.config import API, UnsupportedField


class Format(BaseModel, extra="forbid", populate_by_name=True):
    type: Literal["text", "json", "schema"]
    schema_: type | dict[str, Any] | None = Field(None, alias="schema")  # "schema" shadows a BaseModel attr

    @model_validator(mode="after")
    def _schema_matches_type(self):
        if self.type == "schema" and self.schema_ is None:
            raise ValueError("type='schema' needs a schema")
        if self.type != "schema" and self.schema_ is not None:
            raise ValueError(f"type={self.type!r} forbids schema")
        return self


def translate(fmt: Format, api: API) -> dict:
    match fmt.type:
        case "text":
            return {}  # the provider default everywhere; messages has no explicit spelling at all
        case "json":
            if api == "messages":
                raise UnsupportedField(api, "response_format",
                                       "anthropic has no schema-less JSON mode — pass a schema instead")
            return {"response_format": {"type": "json_object"}}
        case "schema":
            return {"output_format": fmt.schema_} if api == "messages" else {"response_format": fmt.schema_}
        case _:
            assert_never(fmt.type)
