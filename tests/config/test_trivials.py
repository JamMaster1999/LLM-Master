"""The trivial four (system, temperature, max_tokens, timeout): goldens + one batched live call per API."""
from typing import get_args

import pytest
from any_llm import AnyLLM
from any_llm.exceptions import ProviderError

from llm_master.config import max_tokens, system, temperature, thinking, timeout
from llm_master.providers import API

APIS = get_args(API)


def test_temperature_same_everywhere():
    for api in APIS:
        assert temperature.translate(0.7, api) == {"temperature": 0.7}


def test_timeout_same_everywhere():
    for api in APIS:
        assert timeout.translate(30.0, api) == {"timeout": 30.0}


def test_max_tokens_renames_on_responses():
    assert max_tokens.translate(500, "completion") == {"max_tokens": 500}
    assert max_tokens.translate(500, "messages") == {"max_tokens": 500}
    assert max_tokens.translate(500, "responses") == {"max_output_tokens": 500}


def test_system_placement():
    assert system.translate("Be brief.", "completion") == {}  # prepended at assembly instead
    assert system.translate("Be brief.", "messages") == {"system": "Be brief."}
    assert system.translate("Be brief.", "responses") == {"instructions": "Be brief."}


# --- live: one roomy call per API carrying all four, then one tiny call proving truncation ---

S = "You are terse. End every reply with the word BANANA."
HELLO = [{"role": "user", "content": "Say hello."}]
STORY = [{"role": "user", "content": "Write a 500 word story about a lighthouse."}]


@pytest.mark.live
async def test_completion_four_accepted_system_obeyed():
    llm = AnyLLM.create("gemini")
    msgs = [{"role": "system", "content": S}, *HELLO]  # completion has no system param: prepend
    r = await llm.acompletion(model="gemini-2.5-flash", messages=msgs,
                              **temperature.translate(0.2, "completion"),
                              **max_tokens.translate(300, "completion"),
                              **timeout.translate(60, "completion"),
                              **thinking.translate("none", "completion"))
    assert "BANANA" in r.choices[0].message.content


@pytest.mark.live
async def test_messages_accepted_system_obeyed():
    # no temperature: claude-sonnet-5 rejects it as deprecated (live-observed)
    llm = AnyLLM.create("anthropic")
    r = await llm.amessages("claude-sonnet-5", HELLO,
                            **max_tokens.translate(300, "messages"),
                            **timeout.translate(60, "messages"),
                            **thinking.translate("none", "messages"),
                            **system.translate(S, "messages"))
    text = "".join(b.text for b in r.content if b.type == "text")
    assert "BANANA" in text


@pytest.mark.live
async def test_responses_accepted_system_obeyed():
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", HELLO,
                             **temperature.translate(0.2, "responses"),
                             **max_tokens.translate(300, "responses"),
                             **timeout.translate(60, "responses"),
                             **thinking.translate("none", "responses"),
                             **system.translate(S, "responses"))
    text = "".join(part.text for item in r.output if item.type == "message"
                   for part in item.content if part.type == "output_text")
    assert "BANANA" in text


@pytest.mark.live
async def test_completion_max_tokens_truncates():
    llm = AnyLLM.create("gemini")
    r = await llm.acompletion(model="gemini-2.5-flash", messages=STORY,
                              **thinking.translate("none", "completion"), **max_tokens.translate(32, "completion"))
    assert r.choices[0].finish_reason == "length", r.choices[0].finish_reason


@pytest.mark.live
async def test_messages_max_tokens_truncates():
    llm = AnyLLM.create("anthropic")
    r = await llm.amessages("claude-sonnet-5", STORY, 32, **thinking.translate("none", "messages"))
    assert r.stop_reason == "max_tokens", r.stop_reason


@pytest.mark.live
async def test_responses_max_tokens_truncates():
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", STORY,
                             **thinking.translate("none", "responses"), **max_tokens.translate(32, "responses"))
    assert r.status == "incomplete" and r.incomplete_details.reason == "max_output_tokens"


@pytest.mark.live
async def test_timeout_enforced_on_all_apis():
    # 1ms expires before any reply; all three dialects surface it as the typed ProviderError
    with pytest.raises(ProviderError):
        await AnyLLM.create("gemini").acompletion(model="gemini-2.5-flash", messages=HELLO,
                                                  **timeout.translate(0.001, "completion"))
    with pytest.raises(ProviderError):
        await AnyLLM.create("anthropic").amessages("claude-sonnet-5", HELLO, 300,
                                                   **timeout.translate(0.001, "messages"))
    with pytest.raises(ProviderError):
        await AnyLLM.create("openai").aresponses("gpt-5.1", HELLO,
                                                 **timeout.translate(0.001, "responses"))
