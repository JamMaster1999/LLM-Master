"""Retries: a transient failure is tried again on the same candidate, anything else is not. See SPEC.md Failures.

The failures are crafted in front of recorded exchanges: providers don't fail on cue.
"""
import pytest
from any_llm.exceptions import InvalidRequestError, ProviderError, RateLimitError
from test_query import ASK, CONFIGURED, LOOPS

from llm_master.types import Model

GREET = [{"role": "user", "content": "Say good morning to Toronto."}]


async def plain(llm, candidate):
    return await llm.query(candidate, GREET), []


async def streamed(llm, candidate):
    s = await llm.query(candidate, ASK, stream=True)
    return s, [c["type"] async for c in s]


async def speech(llm, candidate):
    return await llm.speech(candidate, "Say: hello there."), []


@pytest.mark.parametrize("name, call, candidate, why", [
    ("anthropic_500_then_ok", plain, CONFIGURED["anthropic"], "Internal server error"),
    ("anthropic_overloaded_then_whole_stream", streamed, LOOPS["anthropic"], "Overloaded"),  # an error event in a 200
    ("anthropic_cut_then_whole_stream", streamed, LOOPS["anthropic"], "ended before"),  # no message_stop
    ("fireworks_cut_then_whole_stream", streamed, LOOPS["fireworks"], "ended before"),  # no closing usage
    ("openai_speech_cut_then_whole", speech, Model(provider="openai", model="gpt-4o-mini-tts"), "ended before"),
], ids=["500", "overloaded-mid-stream", "cut-stream", "completion-without-closing-usage", "cut-speech"])
async def test_a_transient_failure_is_tried_again(name, call, candidate, why, recorded, llm):
    recorded(name)
    made, chunks = await call(llm, candidate)
    if chunks:  # the failed try's output is reset, and the next streams from the start
        made = made.response
        reset = chunks.index("reset-step")
        assert chunks[0] == chunks[reset + 1] == "start-step" and chunks[-1] == "finish-step"
    [attempt] = made.attempts
    [error] = attempt.errors
    assert isinstance(error, ProviderError) and why in str(error)


async def test_a_rate_limit_waits_for_its_retry_after(recorded, llm):
    recorded("anthropic_429_then_ok")  # Retry-After: 1
    r = await llm.query(CONFIGURED["anthropic"], GREET)
    assert isinstance(r.attempts[0].errors[0], RateLimitError)
    assert r.latency - r.attempts[0].latency >= 0.9, "the second try started before Retry-After passed"


async def test_a_bad_request_is_not_tried_again(recorded, llm):
    recorded("anthropic_400")  # one exchange: teardown fails on a second try
    with pytest.raises(InvalidRequestError):
        await llm.query(CONFIGURED["anthropic"], GREET)


async def test_the_fifth_failure_raises(recorded, llm):
    recorded("anthropic_429_five_times")  # Retry-After: 0, five times: teardown fails on a sixth try
    with pytest.raises(RateLimitError):
        await llm.query(CONFIGURED["anthropic"], GREET)
