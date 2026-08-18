"""tool_choice translation: golden shapes for all three APIs, live obedience checks."""
import pytest
from any_llm import AnyLLM
from pydantic import ValidationError

from llm_master.config import thinking, tool_choice, tools
from llm_master.config.tool_choice import ToolChoice


def test_bare_mode_passes_through():
    assert tool_choice.translate(ToolChoice(mode="required"), "completion") == {"tool_choice": "required"}
    assert tool_choice.translate(ToolChoice(mode="none"), "responses") == {"tool_choice": "none"}
    assert tool_choice.translate(ToolChoice(mode="required"), "messages") == {"tool_choice": {"type": "any"}}
    assert tool_choice.translate(ToolChoice(mode="none"), "messages") == {"tool_choice": {"type": "none"}}


def test_required_single_tool():
    c = ToolChoice(mode="required", only="f")
    assert tool_choice.translate(c, "completion") == {"tool_choice": {"type": "function", "function": {"name": "f"}}}
    assert tool_choice.translate(c, "responses") == {"tool_choice": {"type": "function", "name": "f"}}
    assert tool_choice.translate(c, "messages") == {"tool_choice": {"type": "tool", "name": "f"}}


def test_required_subset():
    c = ToolChoice(mode="required", only=["a", "b"])
    assert tool_choice.translate(c, "responses") == {
        "tool_choice": {"type": "allowed_tools", "mode": "required",
                        "tools": [{"type": "function", "name": "a"}, {"type": "function", "name": "b"}]}}
    assert tool_choice.translate(c, "completion") == {
        "tool_choice": {"type": "allowed_tools",
                        "allowed_tools": {"mode": "required",
                                          "tools": [{"type": "function", "function": {"name": "a"}},
                                                    {"type": "function", "function": {"name": "b"}}]}}}
    assert tool_choice.translate(c, "messages") == {"tool_choice": {"type": "any"}}  # anthropic has no subset form


def test_singleton_list_is_its_name():
    one, lst = ToolChoice(mode="required", only="f"), ToolChoice(mode="required", only=["f"])
    for api in ("completion", "messages", "responses"):
        assert tool_choice.translate(lst, api) == tool_choice.translate(one, api)
    with pytest.raises(ValidationError):
        ToolChoice(only=[])


def test_restrict_without_forcing():
    c = ToolChoice(only=["a"])
    assert tool_choice.translate(c, "responses") == {
        "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "a"}]}}
    assert tool_choice.translate(c, "messages") == {"tool_choice": {"type": "auto"}}


def test_parallel_placement():
    assert tool_choice.translate(ToolChoice(parallel=False), "completion") == {"parallel_tool_calls": False}
    assert tool_choice.translate(ToolChoice(parallel=False), "messages") == {
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    assert tool_choice.translate(ToolChoice(mode="required", only="f", parallel=False), "responses") == {
        "parallel_tool_calls": False, "tool_choice": {"type": "function", "name": "f"}}


def test_empty_says_nothing_and_none_forbids_only():
    assert tool_choice.translate(ToolChoice(), "messages") == {}
    assert tool_choice.translate(ToolChoice(), "completion") == {}
    with pytest.raises(ValidationError):
        ToolChoice(mode="none", only="f")
    with pytest.raises(ValidationError):
        ToolChoice(mode="none", parallel=False)  # anthropic's none variant takes no parallel key


# --- live: require get_time on a weather question — only obedience proves the param landed ---

Q = [{"role": "user", "content": "What's the weather in Toronto?"}]

# both target shapes come from the tools translation, so these live tests prove it too
_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                      "parameters": _SCHEMA}},
    {"type": "function", "function": {"name": "get_time", "description": "Current time for a city",
                                      "parameters": _SCHEMA}},
    {"type": "function", "function": {"name": "get_news", "description": "Top headlines for a city",
                                      "parameters": _SCHEMA}},
]
COMPLETION_TOOLS = tools.translate(OPENAI_TOOLS, "completion")["tools"]
RESPONSES_TOOLS = tools.translate(OPENAI_TOOLS, "responses")["tools"]
MESSAGES_TOOLS = tools.translate(OPENAI_TOOLS, "messages")["tools"]


@pytest.mark.live
async def test_responses_required_single_obeyed():
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", Q, tools=RESPONSES_TOOLS, max_output_tokens=1000,
                             **tool_choice.translate(ToolChoice(mode="required", only="get_time"), "responses"))
    called = [item.name for item in r.output if item.type == "function_call"]
    assert called and set(called) == {"get_time"}, f"required get_time, got {called}"


@pytest.mark.live
async def test_responses_required_subset_obeyed():
    # a weather question with get_weather cut from the allowed set: obedience = it stays uncalled
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", Q, tools=RESPONSES_TOOLS, max_output_tokens=1000,
                             **tool_choice.translate(ToolChoice(mode="required", only=["get_time", "get_news"]),
                                                     "responses"))
    called = [item.name for item in r.output if item.type == "function_call"]
    assert called and "get_weather" not in called, f"menu excluded get_weather, got {called}"


@pytest.mark.live
async def test_completion_gemini_required_single_obeyed():
    llm = AnyLLM.create("gemini")
    r = await llm.acompletion(model="gemini-2.5-flash", messages=Q, tools=COMPLETION_TOOLS, max_tokens=1000,
                              **thinking.translate("none", "completion"),
                              **tool_choice.translate(ToolChoice(mode="required", only="get_time"), "completion"))
    called = [tc.function.name for tc in (r.choices[0].message.tool_calls or [])]
    assert called and set(called) == {"get_time"}, f"required get_time, got {called}"


@pytest.mark.live
async def test_completion_gemini_required_subset_obeyed():
    llm = AnyLLM.create("gemini")
    r = await llm.acompletion(model="gemini-2.5-flash", messages=Q, tools=COMPLETION_TOOLS, max_tokens=1000,
                              **thinking.translate("none", "completion"),
                              **tool_choice.translate(ToolChoice(mode="required", only=["get_time", "get_news"]),
                                                      "completion"))
    called = [tc.function.name for tc in (r.choices[0].message.tool_calls or [])]
    assert called and "get_weather" not in called, f"menu excluded get_weather, got {called}"


@pytest.mark.live
async def test_messages_required_single_obeyed():
    llm = AnyLLM.create("anthropic")
    r = await llm.amessages("claude-sonnet-5", Q, 1000, tools=MESSAGES_TOOLS,
                            **thinking.translate("none", "messages"),
                            **tool_choice.translate(ToolChoice(mode="required", only="get_time"), "messages"))
    called = [b.name for b in r.content if b.type == "tool_use"]
    # anthropic guarantees inclusion, not exclusivity (live-observed: it may add parallel calls)
    assert "get_time" in called, f"required get_time, got {called}"


@pytest.mark.live
async def test_messages_required_plus_no_parallel_is_exclusive():
    llm = AnyLLM.create("anthropic")
    r = await llm.amessages("claude-sonnet-5", Q, 1000, tools=MESSAGES_TOOLS,
                            **thinking.translate("none", "messages"),
                            **tool_choice.translate(
                                ToolChoice(mode="required", only="get_time", parallel=False), "messages"))
    called = [b.name for b in r.content if b.type == "tool_use"]
    assert called == ["get_time"], f"required + parallel=False must yield exactly one call, got {called}"


@pytest.mark.live
async def test_messages_required_subset_becomes_any():
    llm = AnyLLM.create("anthropic")
    r = await llm.amessages("claude-sonnet-5", Q, 1000, tools=MESSAGES_TOOLS,
                            **thinking.translate("none", "messages"),
                            **tool_choice.translate(
                                ToolChoice(mode="required", only=["get_weather", "get_time"]), "messages"))
    called = [b.name for b in r.content if b.type == "tool_use"]
    assert called, "any must force some tool call"
