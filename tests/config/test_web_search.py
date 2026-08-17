"""web_search: stance goldens + live search on all three tool providers (card 10)."""
import pytest
from any_llm import AnyLLM

from llm_master.config import UnsupportedField, merge, thinking, web_search
from llm_master.config.web_search import Search


def test_stances_gate():
    with pytest.raises(UnsupportedField):
        web_search.translate(Search(), "completion", "fireworks")


def test_tool_spellings():
    assert web_search.translate(Search(max_searches=2), "messages", "anthropic") == {
        "tools": [{"type": "web_search_20260318", "name": "web_search", "max_uses": 2}]}
    assert web_search.translate(Search(max_searches=2), "responses", "openai") == {
        "tools": [{"type": "web_search"}], "max_tool_calls": 2}
    assert web_search.translate(Search(), "completion", "gemini") == {"tools": [{"google_search": {}}]}


Q = [{"role": "user", "content": "What was the S&P 500's closing value last Friday?"}]


@pytest.mark.live
async def test_completion_gemini_grounded():
    llm = AnyLLM.create("gemini")
    r = await llm.acompletion(model="gemini-2.5-flash", messages=Q, max_tokens=2000,
                              **thinking.translate("none", "completion"),
                              **web_search.translate(Search(), "completion", "gemini"))
    assert r.choices[0].message.content


@pytest.mark.live
async def test_messages_anthropic_searches():
    llm = AnyLLM.create("anthropic")
    r = await llm.amessages("claude-sonnet-5", Q, 2000,
                            **thinking.translate("none", "messages"),
                            **web_search.translate(Search(max_searches=2), "messages", "anthropic"))
    blocks = {b.type for b in r.content}
    assert "server_tool_use" in blocks, blocks
    assert r.usage.server_tool_use.web_search_requests >= 1


@pytest.mark.live
async def test_responses_openai_searches():
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", Q,
                             **thinking.translate("none", "responses"),
                             **web_search.translate(Search(max_searches=2), "responses", "openai"))
    assert any(item.type == "web_search_call" for item in r.output), [i.type for i in r.output]
