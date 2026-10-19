"""The rate limiter: LLM(per_minute=n) lets n tries start in any minute per model, across every call. See SPEC.md
Rate limits."""
import asyncio

import pytest
from cases import CONFIGURED, GREET

from llm_master import limiter
from llm_master.facade import LLM


@pytest.fixture
async def capped():
    llm = LLM(per_minute=1, update_prices=False)
    yield llm
    await llm.aclose()


async def test_a_full_window_holds_the_next_try_before_it_starts(recorded, monkeypatch, capped):
    recorded("gemini_configured")
    monkeypatch.setattr(limiter, "WINDOW", 0.5)
    await capped.limiter.wait("gemini", CONFIGURED["gemini"].model)  # an earlier try took the slot
    r = await capped.query(CONFIGURED["gemini"], GREET)
    [attempt] = r.attempts  # the wait comes before the try: the call's latency counts it, the attempt's doesn't
    assert r.latency - attempt.latency >= 0.45, "the try started inside the earlier one's window"


async def test_each_model_has_its_own_window(recorded, capped):
    recorded("gemini_configured")
    await capped.limiter.wait("gemini", "another-model")
    await asyncio.wait_for(capped.query(CONFIGURED["gemini"], GREET), 5)  # a shared window would hold it 60 s


def test_a_cap_below_one_raises():
    with pytest.raises(ValueError):
        LLM(per_minute=0, update_prices=False)
