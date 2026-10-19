"""Fallback: candidates in order, each with its full retry budget. See SPEC.md Failures.

Most failures are crafted in front of recorded exchanges: providers don't fail on cue.
"""
import genai_prices
import pytest
from any_llm.exceptions import InvalidRequestError, RateLimitError
from test_media import DRAW, OPENAI
from test_query import CONFIGURED, CUT, WRITE

from llm_master.io.response import CutCall
from llm_master.media import MissingMedia
from llm_master.types import Config, Model

GREET = [{"role": "user", "content": "Say good morning to Toronto."}]
CHAIN = [CONFIGURED["anthropic"], CONFIGURED["openai"]]


async def test_an_image_the_first_candidate_cant_make_comes_from_the_next(recorded, llm):
    recorded("gemini_no_image_then_openai")  # gemini_no_image's reply, then openai_image's
    made = await llm.image([Model(provider="gemini", model="gemini-3-pro-image"), OPENAI], DRAW)
    first, second = made.attempts
    assert [type(e) for e in first.errors] == [MissingMedia] and not second.errors
    assert made.origin.provider == "openai"
    tokens = genai_prices.Usage(input_tokens=17, output_tokens=196, output_image_tokens=196)
    assert made.cost == pytest.approx(float(genai_prices.calc_price(tokens, "gpt-image-2", provider_id="openai")
                                            .total_price)), "priced as the answering candidate"


async def test_a_candidate_out_of_tries_hands_off_with_its_errors(recorded, llm):
    recorded("anthropic_429_five_times_then_openai")  # Retry-After: 0
    r = await llm.query(CHAIN, GREET)
    first, second = r.attempts
    assert [type(e) for e in first.errors] == [RateLimitError] * 5 and not second.errors
    assert r.message.origin and r.message.origin.provider == "openai" and "bonjour" in r.message.content.lower()


async def test_a_stream_that_fails_over_starts_a_new_step(recorded, llm):
    recorded("anthropic_cut_call_then_openai_stream")  # a CutCall isn't retried: the next candidate streams
    fallback = Model(provider="openai", model="gpt-5.6-luna", config=Config(tools=[WRITE], thinking="none"))
    s = await llm.query([CUT["anthropic"], fallback],
                        [{"role": "user", "content": "Call write_file with a 300-word poem as content."}], stream=True)
    chunks = [c["type"] async for c in s]
    reset = chunks.index("reset-step")
    assert chunks[0] == chunks[reset + 1] == "start-step" and chunks[-1] == "finish-step"
    assert s.response and s.response.message.calls
    first, second = s.response.attempts
    assert [type(e) for e in first.errors] == [CutCall] and not second.errors


async def test_when_every_candidate_fails_the_last_error_raises(recorded, llm):
    recorded("anthropic_400_then_openai_400")
    with pytest.raises(InvalidRequestError, match="max_output_tokens") as caught:
        await llm.query(CHAIN, GREET)
    [note] = caught.value.__notes__
    assert note.startswith("earlier, anthropic/claude-haiku-5-5: InvalidRequestError")


async def test_a_call_with_no_candidate_sends_nothing(recorded, llm):
    recorded("nothing_sent")
    with pytest.raises(ValueError):
        await llm.query([], GREET)


@pytest.mark.live
async def test_a_model_the_provider_lacks_hands_off(llm):
    r = await llm.query([Model(provider="openai", model="gpt-no-such-model"),
                         Model(provider="gemini", model="gemini-3.1-flash-lite")], GREET)
    first, _ = r.attempts
    assert len(first.errors) == 1 and r.message.origin and r.message.origin.provider == "gemini"
