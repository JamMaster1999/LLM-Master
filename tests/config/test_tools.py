"""tools definitions: goldens for both input kinds, live proof of gemini's converter path."""
import pytest
from any_llm import AnyLLM
from pydantic import ValidationError

from llm_master.config import thinking, tool_choice, tools
from llm_master.config.tool_choice import ToolChoice
from llm_master.types import Config

OPENAI_TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def get_time(city: str) -> str:
    """Current time for a city."""
    return ""


def test_dicts_reshape_per_api():
    assert tools.translate(OPENAI_TOOLS, "completion") == {"tools": OPENAI_TOOLS}
    assert tools.translate(OPENAI_TOOLS, "messages") == {"tools": [{
        "name": "get_weather", "description": "Current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]}
    assert tools.translate(OPENAI_TOOLS, "responses") == {"tools": [{
        "type": "function", "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]}


def test_callables_convert_then_reshape():
    for api, name_key in (("messages", "input_schema"), ("responses", "parameters")):
        out = tools.translate([get_time], api)["tools"][0]
        assert out["name"] == "get_time"
        assert out["description"]
        assert "city" in out[name_key]["properties"]


def test_unknown_keys_ride_through():
    strict = [{"type": "function", "function": {"name": "f", "parameters": {}, "strict": True}}]
    assert tools.translate(strict, "responses")["tools"] == [
        {"type": "function", "name": "f", "parameters": {}, "strict": True}]
    assert tools.translate(strict, "messages")["tools"] == [
        {"name": "f", "strict": True, "input_schema": {}}]  # anthropic judges strict, not us


def test_provider_hosted_tools_fail_validation():
    Config(tools=[get_time, *OPENAI_TOOLS])
    for hosted in ({"type": "web_search"}, {"type": "function"}, "web_search", object()):
        with pytest.raises(ValidationError, match="function tools only"):
            Config(tools=[get_time, hosted])


@pytest.mark.live
async def test_completion_gemini_calls_translated_tool():
    llm = AnyLLM.create("gemini")
    r = await llm.acompletion(model="gemini-2.5-flash",
                              messages=[{"role": "user", "content": "What's the weather in Toronto?"}],
                              max_tokens=500,
                              **thinking.translate("none", "completion"),
                              **tools.translate(OPENAI_TOOLS, "completion"),
                              **tool_choice.translate(ToolChoice(mode="required"), "completion"))
    calls = r.choices[0].message.tool_calls
    assert calls and calls[0].function.name == "get_weather"
