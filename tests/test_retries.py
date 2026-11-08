"""Retries: a transient failure is tried again on the same candidate, anything else is not. See SPEC.md Failures.

The failures are crafted in front of recorded exchanges: providers don't fail on cue.
"""
import gc
import json
import types

import pytest
from any_llm.exceptions import AnyLLMError, InvalidRequestError, ProviderError, RateLimitError
from cases import ASK, CONFIGURED, GREET, LOOPS
from conftest import cut, exchanges, failing, paced

from llm_master import pipeline
from llm_master.types import Model

SERVER_ERROR = {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
RATE_LIMITED = {"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}}
GEMINI_RATE_LIMITED = {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED",
                                 "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}]}}
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
    ((failing("anthropic_configured", 429, RATE_LIMITED), "anthropic_configured"), plain, CONFIGURED["anthropic"],
     "Rate limited"),  # no Retry-After
    ((failed("openai_configured", "server_error"), "openai_configured"), plain, CONFIGURED["openai"], "server_error"),
    ((cut("anthropic_loop_stream", "content_block_start", OVERLOADED), exchanges("anthropic_loop_stream")[:1]),
     streamed, LOOPS["anthropic"], "Overloaded"),  # an error event inside a 200 stream
    ((cut("fireworks_loop_stream", '"usage":{'), exchanges("fireworks_loop_stream")[:1]),
     streamed, LOOPS["fireworks"], "ended before"),  # no closing usage
    ((cut("openai_speech", "speech.audio.done"), "openai_speech"), speech, SPEECH, "ended before"),
    ((paced(exchanges("gemini_configured"), 60), "gemini_configured"), plain, CONFIGURED["gemini"], "nothing came"),
    ((paced(exchanges("gemini_loop_stream")[:1], 60), exchanges("gemini_loop_stream")[:1]), streamed, LOOPS["gemini"],
     "nothing came"),  # genai sets no timeout: ours ends a quiet try
], ids=["500", "429-without-retry-after", "openai-failed-server-error", "overloaded-mid-stream",
        "completion-without-closing-usage", "cut-speech", "quiet-gemini", "quiet-gemini-stream"])
async def test_a_transient_failure_is_tried_again(steps, call, candidate, why, recorded, monkeypatch, llm):
    monkeypatch.setattr(pipeline, "QUIET", 1.0)
    recorded(*steps)
    llm.client(candidate.provider)  # built ahead: a cold build counts against the quiet rows' timer
    made = await call(llm, candidate)
    [attempt] = made.attempts
    [error] = attempt.errors
    assert isinstance(error, AnyLLMError) and why in str(error)


def bodies(root: object, text: str) -> list[bytes]:
    """The bytes reachable from root that hold text: an encoded request, say."""
    found, seen, todo = [], set(), [root]
    while todo:
        o = todo.pop()
        if id(o) in seen or isinstance(o, (type, types.ModuleType)):
            continue
        seen.add(id(o))
        if isinstance(o, bytes) and text.encode() in o:
            found.append(o)
        todo += gc.get_referents(o)
    return found


async def test_a_kept_error_holds_no_copy_of_its_request(recorded, llm):
    recorded(failing("anthropic_configured", 500, SERVER_ERROR), "anthropic_configured")
    r = await llm.query(CONFIGURED["anthropic"], GREET)
    [error] = r.attempts[0].errors
    assert not bodies(error, GREET[0]["content"]), "the error kept the request body"


async def test_a_retried_stream_resets_and_starts_a_new_step(recorded, llm):
    recorded(cut("anthropic_loop_stream", "message_delta"), exchanges("anthropic_loop_stream")[:1])
    chunks = [c["type"] async for c in await llm.query(LOOPS["anthropic"], ASK, stream=True)]
    reset = chunks.index("reset-step")
    assert chunks[0] == chunks[reset + 1] == "start-step" and chunks[-1] == "finish-step"


@pytest.mark.parametrize("steps, candidate", [
    ((failing("anthropic_configured", 429, RATE_LIMITED, headers={"retry-after": "1"}), "anthropic_configured"),
     CONFIGURED["anthropic"]),
    ((failing("gemini_configured", 429, GEMINI_RATE_LIMITED), "gemini_configured"), CONFIGURED["gemini"]),
], ids=["retry-after", "gemini-retry-delay"])
async def test_a_rate_limit_waits_for_its_retry_after(steps, candidate, recorded, llm):
    recorded(*steps)
    r = await llm.query(candidate, GREET)
    assert isinstance(r.attempts[0].errors[0], RateLimitError)
    # full jitter waits half a second at most before the second try
    assert r.latency - r.attempts[0].latency >= 0.9, "the second try started before the provider's delay passed"


async def test_waits_past_the_budget_end_the_candidate(recorded, monkeypatch, llm):
    monkeypatch.setattr(pipeline, "BUDGET", 0.5)
    # 0.3 + 0.3 is past the budget: teardown fails on a third try, or on no second
    recorded(failing("anthropic_configured", 429, RATE_LIMITED, times=2, headers={"retry-after": "0.3"}))
    with pytest.raises(RateLimitError):
        await llm.query(CONFIGURED["anthropic"], GREET)


async def test_a_genai_stream_that_keeps_coming_outlasts_the_timer(recorded, monkeypatch, llm):
    monkeypatch.setattr(pipeline, "QUIET", 1.0)
    recorded(paced(exchanges("gemini_loop_stream")[:1], 0.3))  # four events, 1.2 seconds in all
    llm.client("gemini")  # built ahead, after the stand-in: its build would count against the first chunk
    r = await streamed(llm, LOOPS["gemini"])
    assert r.attempts[0].errors == [], "the timer cut a stream that kept sending"


@pytest.mark.parametrize("steps, candidate", [
    (failing("anthropic_configured", 400, BAD_REQUEST), CONFIGURED["anthropic"]),
    (failed("openai_configured", "invalid_prompt"), CONFIGURED["openai"]),  # a failed status that fails again
], ids=["400", "openai-failed-invalid-prompt"])
async def test_a_failure_that_would_repeat_is_not_tried_again(steps, candidate, recorded, llm):
    recorded(steps)  # one exchange: teardown fails on a second try
    with pytest.raises(InvalidRequestError):
        await llm.query(candidate, GREET)


async def test_the_fifth_failure_raises(recorded, llm):
    recorded(failing("anthropic_configured", 429, RATE_LIMITED, times=5, headers={"retry-after": "0"}))
    with pytest.raises(RateLimitError):  # teardown fails on a sixth try
        await llm.query(CONFIGURED["anthropic"], GREET)
