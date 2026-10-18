"""Live: the thinking translation drives real reasoning on and off, per API."""
import pytest
from any_llm import AnyLLM
from any_llm.types.messages import MessageResponse

from llm_master.config import thinking

Q = "What is 15% of 240? Think it through carefully before answering."
RIDDLE = [{"role": "user", "content": "A farmer has 17 sheep; all but 9 run away, then he buys twice "
           "as many as remain and sells a third. How many now? Reason carefully."}]


def test_responses_levels_carry_summary_ask():
    assert thinking.translate("high", "responses") == {"reasoning": {"effort": "high", "summary": "auto"}}
    assert thinking.translate("none", "responses") == {"reasoning": {"effort": "none"}}


def test_messages_levels_spelling():
    # the live test can't catch a lost output_config: adaptive alone still produces thinking blocks
    assert thinking.translate("low", "messages") == {"thinking": {"type": "adaptive", "display": "summarized"},
                                                      "output_config": {"effort": "low"}}
    assert thinking.translate("none", "messages") == {"thinking": {"type": "disabled"}}


@pytest.mark.live
async def test_responses_levels_always_return_summary_text():
    # needs real reasoning: openai emits summaries only when there is enough to summarize
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", RIDDLE, max_output_tokens=4000, **thinking.translate("high", "responses"))
    text = "".join(part.text for item in r.output if item.type == "reasoning"
                   for part in (item.summary or []))
    assert text, "levels must always fetch readable thought text"


@pytest.mark.live
async def test_completion_gemini():
    llm = AnyLLM.create("gemini")

    def thought(r):  # our fork reports gemini thoughts in completion_tokens_details (unset when zero)
        d = r.usage.completion_tokens_details
        return d.reasoning_tokens if d else 0

    low = await llm.acompletion(model="gemini-2.5-flash", messages=[{"role": "user", "content": Q}],
                                max_tokens=3000, **thinking.translate("low", "completion"))
    none = await llm.acompletion(model="gemini-2.5-flash", messages=[{"role": "user", "content": Q}],
                                 max_tokens=3000, **thinking.translate("none", "completion"))
    assert thought(low) > 0, f"low produced no thinking tokens: {low.usage}"
    assert thought(none) == 0, f"none still thought: {none.usage}"


@pytest.mark.live
async def test_messages_anthropic():
    llm = AnyLLM.create("anthropic")
    low = await llm.amessages("claude-sonnet-5", [{"role": "user", "content": Q}], 3000,
                              **thinking.translate("low", "messages"))
    none = await llm.amessages("claude-sonnet-5", [{"role": "user", "content": Q}], 3000,
                               **thinking.translate("none", "messages"))
    assert isinstance(low, MessageResponse) and isinstance(none, MessageResponse)
    # readable text, not just a block: Claude 5.x returns empty thinking unless display is summarized
    assert any(b.type == "thinking" and b.thinking for b in low.content), [b.type for b in low.content]
    assert not any(b.type == "thinking" for b in none.content)


@pytest.mark.live
async def test_responses_openai():
    llm = AnyLLM.create("openai")
    low = await llm.aresponses("gpt-5.1", [{"role": "user", "content": Q}],
                               max_output_tokens=3000, **thinking.translate("low", "responses"))
    none = await llm.aresponses("gpt-5.1", [{"role": "user", "content": Q}],
                                max_output_tokens=3000, **thinking.translate("none", "responses"))
    assert low.usage.output_tokens_details.reasoning_tokens > 0
    assert none.usage.output_tokens_details.reasoning_tokens == 0
