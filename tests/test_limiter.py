"""The rate limiter: LLM(per_minute=n) lets n tries start in any minute per model, across every call. See SPEC.md
Rate limits."""
import asyncio
import json
import time

import pytest
from cases import CONFIGURED, GREET
from conftest import FIXTURES

from llm_master import limiter
from llm_master.facade import LLM
from llm_master.types import Model


@pytest.fixture
async def capped():
    llm = LLM(per_minute=1, update_prices=False)
    yield llm
    await llm.aclose()


@pytest.fixture
def answered(provider_server):
    """Every provider answers with gemini_configured's recorded reply; returns when each request arrived, and its
    prompt."""
    reply = json.loads((FIXTURES / "gemini_configured.json").read_text())[0]["response"]
    hits: list[tuple[float, str]] = []

    def answer(handler, provider, path, body):
        hits.append((time.monotonic(), json.loads(body)["contents"][-1]["parts"][0]["text"]))
        payload = reply["body"].encode()
        handler.send_response(reply["status"])
        handler.send_header("Content-Type", reply["type"])
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    provider_server(answer)
    return hits


async def test_a_full_window_holds_each_try_until_a_slot_frees(answered, monkeypatch, capped):
    monkeypatch.setattr(limiter, "WINDOW", 0.5)
    capped.client("gemini")  # built ahead, so each try reaches the server as it starts
    prompts = [f"Say {n}." for n in range(3)]
    made = await asyncio.gather(*(capped.query(CONFIGURED["gemini"], [{"role": "user", "content": p}])
                                  for p in prompts))
    assert [prompt for _, prompt in answered] == prompts, "the tries started out of arrival order"
    starts = [at for at, _ in answered]
    # 0.1 s of slack: a request's trip to the server varies; a broken window lets one through at once
    assert all(b - a >= 0.4 for a, b in zip(starts, starts[1:])), f"a try started inside the window: {starts}"
    [attempt] = made[-1].attempts  # the wait comes before the try: the call's latency counts it, the attempt's doesn't
    assert made[-1].latency - attempt.latency >= 0.9, "the last call didn't wait two windows"


async def test_each_model_has_its_own_window(answered, capped):
    other = Model(provider="gemini", model="another-model", config=CONFIGURED["gemini"].config)
    calls = asyncio.gather(capped.query(CONFIGURED["gemini"], GREET), capped.query(other, GREET))
    await asyncio.wait_for(calls, 5)  # a shared window would hold the second call 60 s
    assert len(answered) == 2


def test_a_cap_below_one_raises():
    with pytest.raises(ValueError):
        LLM(per_minute=0, update_prices=False)
