"""response_format: what shape the output must be. See CONFIG.md.

Flat like ToolChoice: type says the mode, schema carries the shape.
Format(type="json") = valid JSON, any shape. Format(type="schema", schema=X) = that shape
exactly, as a pydantic class or a json-schema dict. Format(type="text") = plain text — every
provider's default, so it emits nothing. None = unset.
"""
from typing import Any, Literal, assert_never

from anthropic import transform_schema
from any_llm.utils.structured_output import get_strict_json_schema
from pydantic import BaseModel, Field, model_validator

from llm_master.config import UnsupportedField
from llm_master.providers import API


class Format(BaseModel, extra="forbid", populate_by_name=True):
    type: Literal["text", "json", "schema"]
    schema_: type | dict[str, Any] | None = Field(default=None, alias="schema")  # "schema" shadows a BaseModel attr

    @model_validator(mode="after")
    def _schema_matches_type(self):
        if self.type == "schema" and self.schema_ is None:
            raise ValueError("type='schema' needs a schema")
        if self.type != "schema" and self.schema_ is not None:
            raise ValueError(f"type={self.type!r} forbids schema")
        return self


def translate(fmt: Format, api: API) -> dict:
    match api:
        case "completion":
            match fmt.type:
                case "text":
                    return {}  # the provider default everywhere; messages has no explicit spelling at all
                case "json":
                    return {"response_format": {"type": "json_object"}}
                case "schema":
                    return {"response_format": {"type": "json_schema", "json_schema": _openai(fmt.schema_)}}
                case _:
                    assert_never(fmt.type)
        case "messages":
            match fmt.type:
                case "text":
                    return {}
                case "json":
                    raise UnsupportedField(api, "response_format",
                                           "anthropic has no schema-less JSON mode — pass a schema instead")
                case "schema":  # output_config, so it merges with thinking's effort instead of colliding
                    schema = transform_schema(fmt.schema_) if isinstance(fmt.schema_, type) else fmt.schema_
                    return {"output_config": {"format": {"type": "json_schema", "schema": schema}}}
                case _:
                    assert_never(fmt.type)
        case "responses":
            match fmt.type:
                case "text":
                    return {}
                case "json":
                    return {"response_format": {"type": "json_object"}}
                case "schema":
                    return {"response_format": {"type": "json_schema", **_openai(fmt.schema_)}}
                case _:
                    assert_never(fmt.type)
        case _:
            assert_never(api)


def _openai(schema: type | dict[str, Any] | None) -> dict[str, Any]:
    """The json_schema fields. A class gets the strict schema openai's own SDK sends for it; a dict rides as written."""
    if isinstance(schema, type):
        return {"name": schema.__name__, "schema": get_strict_json_schema(schema), "strict": True}
    return {"name": "response", "schema": schema}
