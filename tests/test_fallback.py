"""Fallback: candidates in order, each with its full retry budget. See SPEC.md Failures.

Most failures are crafted in front of recorded exchanges: providers don't fail on cue.
"""
import genai_prices
import pytest
from any_llm.exceptions import InvalidRequestError, RateLimitError
from cases import BAD_REQUEST, CHAIN, DRAW, GREET, OPENAI_IMAGE, RATE_LIMITED
from conftest import exchanges, failing

from llm_master import pipeline
from llm_master.config import UnsupportedField
from llm_master.media import Media, MissingMedia
from llm_master.types import Config, Model

OPENAI_RATE_LIMITED = {"error": {"message": "Rate limited", "type": "requests", "param": None,
                                 "code": "rate_limit_exceeded"}}


async def test_an_image_the_first_candidate_cant_make_comes_from_the_next(recorded, llm):
    recorded("gemini_no_image", "openai_image")
    made = await llm.image([Model(provider="gemini", model="gemini-3-pro-image"), OPENAI_IMAGE], DRAW)
    first, second = made.attempts
    assert [type(e) for e in first.errors] == [MissingMedia] and not second.errors
    assert made.origin.provider == "openai"
    tokens = genai_prices.Usage(input_tokens=17, output_tokens=196, output_image_tokens=196)
    assert made.cost == pytest.approx(float(genai_prices.calc_price(tokens, "gpt-image-2", provider_id="openai")
                                            .total_price)), "priced as the answering candidate"


async def test_a_media_candidate_with_config_it_cant_take_fails_alone(recorded, llm):
    recorded(exchanges("openai_image"))  # test_media's recording: RECORD=1 leaves it to that test
    made = await llm.image([OPENAI_IMAGE.model_copy(update={"config": Config(system="x")}), OPENAI_IMAGE], DRAW)
    first, second = made.attempts
    assert [type(e) for e in first.errors] == [UnsupportedField] and not second.errors


async def test_a_candidate_out_of_tries_hands_off_with_its_errors(recorded, llm):
    recorded(failing("anthropic_configured", 429, RATE_LIMITED, times=5, headers={"retry-after": "0"}),
             "openai_configured")
    r = await llm.query(CHAIN, GREET)
    first, second = r.attempts
    assert [type(e) for e in first.errors] == [RateLimitError] * 5 and not second.errors
    assert r.message.origin and r.message.origin.provider == "openai" and "bonjour" in r.message.content.lower()


async def test_each_candidate_gets_its_own_wait_budget(recorded, monkeypatch, llm):
    monkeypatch.setattr(pipeline, "BUDGET", 0.5)
    # the first spends 0.3 of its 0.5 and stops; a shared budget would stop the second at its first 0.3 wait
    recorded(failing("anthropic_configured", 429, RATE_LIMITED, times=2, headers={"retry-after": "0.3"}),
             failing("openai_configured", 429, OPENAI_RATE_LIMITED, headers={"retry-after": "0.3"}),
             "openai_configured")
    r = await llm.query(CHAIN, GREET)
    first, second = r.attempts
    assert (len(first.errors), len(second.errors)) == (2, 1)


async def test_when_every_candidate_fails_the_last_error_raises(recorded, llm):
    recorded(failing("anthropic_configured", 400, BAD_REQUEST),
             failing("openai_configured", 400, {"error": {
                 "message": "Invalid 'max_output_tokens': too large.", "type": "invalid_request_error",
                 "param": "max_output_tokens", "code": None}}))
    with pytest.raises(InvalidRequestError, match="max_output_tokens") as caught:
        await llm.query(CHAIN, GREET)
    [note] = caught.value.__notes__
    assert note.startswith("earlier, anthropic/claude-haiku-5-5: InvalidRequestError")


async def test_a_call_with_no_candidate_sends_nothing(recorded, llm):
    recorded("nothing_sent")
    with pytest.raises(ValueError):
        await llm.query([], GREET)


async def plain(llm, candidates):
    return await llm.query(candidates, GREET)


async def streamed(llm, candidates):
    s = await llm.query(candidates, GREET, stream=True)
    assert [c["type"] async for c in s][-1] == "finish-step"
    return s.response


async def image(llm, candidates):
    return await llm.image(candidates, DRAW)


async def speech(llm, candidates):
    return await llm.speech(candidates, "Say: hello there.")


LITE = Model(provider="gemini", model="gemini-3.1-flash-lite")


@pytest.mark.live
@pytest.mark.parametrize("call, candidates", [
    (plain, [Model(provider="openai", model="gpt-no-such-model"), LITE]),
    (streamed, [Model(provider="anthropic", model="claude-no-such-model"), LITE]),
    (image, [Model(provider="gemini", model="gemini-no-such-image"), OPENAI_IMAGE]),  # gemini's filter is erratic
    (speech, [Model(provider="gemini", model="gemini-no-such-tts"), Model(provider="openai", model="gpt-4o-mini-tts")]),
], ids=["query", "stream", "image", "speech"])
async def test_a_model_the_provider_lacks_hands_off(call, candidates, llm):
    made = await call(llm, candidates)
    first, second = made.attempts
    assert len(first.errors) == 1, f"a model the provider lacks was tried again: {first.errors}"
    assert (second.provider, second.model) == (candidates[1].provider, candidates[1].model)
    assert made.data if isinstance(made, Media) else made.message.content, "the next candidate answered nothing"
