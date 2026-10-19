"""Telemetry: one PostHog event per call, against a stand-in PostHog. See SPEC.md Telemetry."""
import json
import logging

import pytest
from cases import CHAIN, CUT, GREET
from conftest import POSTHOG_KEY, failing, local_server

from llm_master.facade import LLM
from llm_master.io.response import CutCall
from llm_master.types import Model

RATE_LIMITED = {"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}}


@pytest.fixture
def posthog(monkeypatch, request) -> list[dict]:
    """Points telemetry at a local PostHog; returns the events it receives."""
    events: list[dict] = []

    def answer(handler, body):
        events.extend(json.loads(body)["batch"])
        handler.send_response(200)
        handler.send_header("Content-Length", "2")
        handler.end_headers()
        handler.wfile.write(b"{}")
    monkeypatch.setenv("POSTHOG_API_KEY", "test")
    monkeypatch.setenv("POSTHOG_HOST", f"http://127.0.0.1:{local_server(request, answer)}")
    return events


async def test_an_answered_call_reports_its_tries_and_spend(posthog, recorded):
    recorded(failing("anthropic_configured", 429, RATE_LIMITED, times=5, headers={"retry-after": "0"}),
             "openai_configured")
    llm = LLM(update_prices=False)
    r = await llm.query(CHAIN, GREET)
    await llm.aclose()  # sends what's queued
    [event] = posthog
    p = event["properties"]
    assert event["event"] == "$ai_generation" and p["$ai_span_name"] == "query"
    assert (p["$ai_provider"], p["$ai_model"]) == ("openai", "gpt-5.6-luna")
    assert (p["failed_tries"], p["fallbacks"]) == (5, 1)
    assert p["errors"] == ["anthropic/claude-haiku-5-5: RateLimitError 429"] * 5
    assert r.usage and (p["$ai_input_tokens"], p["$ai_output_tokens"]) == (r.usage.input, r.usage.output)
    assert (p["$ai_total_cost_usd"], p["$ai_latency"], p["$ai_stop_reason"]) == (r.cost, r.latency, "stop")


async def test_no_content_reaches_posthog(posthog, recorded):
    recorded("openai_cut_call")  # its arguments, cut at '{"content":"At', land in the CutCall's message
    llm = LLM(update_prices=False)
    with pytest.raises(CutCall):
        await llm.query(CUT["openai"], [{"role": "user", "content": "Call write_file with a 300-word poem as content."}])
    await llm.aclose()
    [event] = posthog
    p = event["properties"]
    assert p["$ai_is_error"] and p["$ai_error"] == "CutCall" and p["errors"] == ["openai/gpt-5.6-luna: CutCall"]
    sent = json.dumps(event, ensure_ascii=False)
    assert "poem" not in sent and "content" not in sent, "content reached PostHog"


@pytest.mark.live
async def test_events_reach_posthog(monkeypatch, caplog):
    assert POSTHOG_KEY is not None, "llm_master/.env has no POSTHOG_API_KEY"  # never print the key itself
    monkeypatch.setenv("POSTHOG_API_KEY", POSTHOG_KEY)
    llm = LLM(update_prices=False)
    await llm.query(Model(provider="gemini", model="gemini-3.1-flash-lite"), GREET)
    with caplog.at_level(logging.WARNING, logger="posthog"):
        await llm.aclose()
    assert not [r.getMessage() for r in caplog.records if r.name.startswith("posthog")]
