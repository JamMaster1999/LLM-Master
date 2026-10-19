"""Retries: a transient failure is tried again on the same candidate, anything else is not. See SPEC.md Failures.

The failures are crafted in front of recorded exchanges: providers don't fail on cue.
"""
import json

import pytest
from any_llm.exceptions import InvalidRequestError, ProviderError, RateLimitError
from cases import ASK, CONFIGURED, GREET, LOOPS
from conftest import cut, exchanges, failing

from llm_master.types import Model

SERVER_ERROR = {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
RATE_LIMITED = {"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}}
BAD_REQUEST = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: too large"}}
OVERLOADED = 'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}\n\n'
SPEECH = Model(provider="openai", model="gpt-4o-mini-tts")


def failed(name: str, code: str) -> list[dict]:
    """name's openai Response, failed with code: a 200 that carries the failure."""
    first = exchanges(name)[0]
    body = json.loads(first["response"]["body"]) | {"status": "failed", "output": [],
                                                    "error": {"code": code, "message": f"{code}, crafted"}}
    return [{**first, "response": {**first["response"], "body": json.dumps(body)}}]


async def plain(llm, candidate):
    return await llm.query(candidate, GREET)


async def streamed(llm, candidate):
    s = await llm.query(candidate, ASK, stream=True)
    async for _ in s:
        pass
    return s.response


async def speech(llm, candidate):
    return await llm.speech(candidate, "Say: hello there.")


@pytest.mark.parametrize("steps, call, candidate, why", [
    ((failing("anthropic_configured", 500, SERVER_ERROR), "anthropic_configured"), plain, CONFIGURED["anthropic"],
     "Internal server error"),
    ((failed("openai_configured", "server_error"), "openai_configured"), plain, CONFIGURED["openai"], "server_error"),
    ((cut("anthropic_loop_stream", "content_block_start", OVERLOADED), exchanges("anthropic_loop_stream")[:1]),
     streamed, LOOPS["anthropic"], "Overloaded"),  # an error event inside a 200 stream
    ((cut("anthropic_loop_stream", "message_delta"), exchanges("anthropic_loop_stream")[:1]),
     streamed, LOOPS["anthropic"], "ended before"),  # no message_stop
    ((cut("fireworks_loop_stream", '"usage":{'), exchanges("fireworks_loop_stream")[:1]),
     streamed, LOOPS["fireworks"], "ended before"),  # no closing usage
    ((cut("openai_speech", "speech.audio.done"), "openai_speech"), speech, SPEECH, "ended before"),
], ids=["500", "openai-failed-server-error", "overloaded-mid-stream", "cut-stream", "completion-without-closing-usage",
        "cut-speech"])
async def test_a_transient_failure_is_tried_again(steps, call, candidate, why, recorded, llm):
    recorded(*steps)
    made = await call(llm, candidate)
    [attempt] = made.attempts
    [error] = attempt.errors
    assert isinstance(error, ProviderError) and why in str(error)


async def test_a_retried_stream_resets_and_starts_a_new_step(recorded, llm):
    recorded(cut("anthropic_loop_stream", "message_delta"), exchanges("anthropic_loop_stream")[:1])
    chunks = [c["type"] async for c in await llm.query(LOOPS["anthropic"], ASK, stream=True)]
    reset = chunks.index("reset-step")
    assert chunks[0] == chunks[reset + 1] == "start-step" and chunks[-1] == "finish-step"


async def test_a_rate_limit_waits_for_its_retry_after(recorded, llm):
    recorded(failing("anthropic_configured", 429, RATE_LIMITED, headers={"retry-after": "1"}), "anthropic_configured")
    r = await llm.query(CONFIGURED["anthropic"], GREET)
    assert isinstance(r.attempts[0].errors[0], RateLimitError)
    assert r.latency - r.attempts[0].latency >= 0.9, "the second try started before Retry-After passed"


@pytest.mark.parametrize("steps, candidate", [
    (failing("anthropic_configured", 400, BAD_REQUEST), CONFIGURED["anthropic"]),
    (failed("openai_configured", "invalid_prompt"), CONFIGURED["openai"]),  # a failed status that fails again
], ids=["400", "openai-failed-invalid-prompt"])
async def test_a_failure_that_would_repeat_is_not_tried_again(steps, candidate, recorded, llm):
    recorded(steps)  # one exchange: teardown fails on a second try
    with pytest.raises(InvalidRequestError):
        await llm.query(candidate, GREET)

