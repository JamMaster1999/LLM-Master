"""The rate limiter: a sliding window per (provider, model), waited on before every try. See SPEC.md Rate limits."""
import asyncio
import time

import pytest
from cases import ASK, CONFIGURED, GREET, LOOPS

from llm_master.config import moderation
from llm_master.providers import PROVIDERS


async def test_a_full_window_holds_a_try_until_its_oldest_start_leaves(monkeypatch, llm):
    monkeypatch.setitem(PROVIDERS["anthropic"]["models"], "m", {"requests": 2, "interval": 1})
    started = time.monotonic()
    await llm.limiter.wait("anthropic", "m")
    await llm.limiter.wait("anthropic", "m")
    await llm.limiter.wait("anthropic", "other")  # another model, another window
    assert time.monotonic() - started < 0.5
    await llm.limiter.wait("anthropic", "m")
    assert time.monotonic() - started >= 1


async def stream(llm):
    async for _ in await llm.query(LOOPS["anthropic"], ASK, stream=True):
        pass


@pytest.mark.parametrize("call, provider, window", [
    (lambda llm: llm.query(LOOPS["anthropic"], ASK), "anthropic", LOOPS["anthropic"].model),
    (stream, "anthropic", LOOPS["anthropic"].model),
    (lambda llm: llm.query(LOOPS["anthropic"], ASK, moderation=True), "openai", moderation.MODEL),
], ids=["plain", "stream", "moderation"])
async def test_a_call_waits_for_its_window_before_it_sends(call, provider, window, recorded, monkeypatch, llm):
    recorded("nothing_sent")  # teardown fails on any request
    monkeypatch.setitem(PROVIDERS[provider]["models"], window, {"requests": 1, "interval": 60})
    await llm.limiter.wait(provider, window)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(call(llm), 0.2)


async def test_the_wait_counts_toward_the_calls_latency_not_the_attempts(recorded, monkeypatch, llm):
    recorded("gemini_configured")
    candidate = CONFIGURED["gemini"]
    monkeypatch.setitem(PROVIDERS["gemini"]["models"], candidate.model, {"requests": 1, "interval": 1})
    await llm.limiter.wait("gemini", candidate.model)
    r = await llm.query(candidate, GREET)
    [attempt] = r.attempts
    assert r.latency - attempt.latency >= 0.9, "the attempt counted the limiter's wait"
