"""Telemetry: one PostHog event per call, against a stand-in PostHog. See SPEC.md Telemetry."""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import pytest
from conftest import POSTHOG_KEY
from test_query import CONFIGURED

from llm_master.facade import LLM
from llm_master.types import Model

GREET = [{"role": "user", "content": "Say good morning to Toronto."}]
CHAIN = [CONFIGURED["anthropic"], CONFIGURED["openai"]]


@pytest.fixture
def posthog(monkeypatch, request) -> list[dict]:
    """Points telemetry at a local PostHog; returns the events it receives."""
    events: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            events.extend(json.loads(self.rfile.read(int(self.headers["Content-Length"])))["batch"])
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format, *args):
            pass

    server = ThreadingTCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    request.addfinalizer(server.server_close)
    request.addfinalizer(server.shutdown)
    monkeypatch.setenv("POSTHOG_API_KEY", "test")
    monkeypatch.setenv("POSTHOG_HOST", f"http://127.0.0.1:{server.server_address[1]}")
    return events


async def test_an_answered_call_reports_its_tries_and_spend_but_no_content(posthog, recorded):
    recorded("anthropic_429_five_times_then_openai")
    llm = LLM(update_prices=False)
    r = await llm.query(CHAIN, GREET)
    await llm.aclose()  # sends what's queued
    [event] = posthog
    p = event["properties"]
    assert event["event"] == "$ai_generation" and p["$ai_span_name"] == "query"
    assert (p["$ai_provider"], p["$ai_model"]) == ("openai", "gpt-5.6-luna")
    assert (p["retries"], p["fallbacks"], len(p["errors"])) == (5, 1, 5)
    assert r.usage and (p["$ai_input_tokens"], p["$ai_output_tokens"]) == (r.usage.input, r.usage.output)
    assert (p["$ai_total_cost_usd"], p["$ai_latency"], p["$ai_stop_reason"]) == (r.cost, r.latency, "stop")
    sent = json.dumps(event, ensure_ascii=False)
    assert "Toronto" not in sent and str(r.message.content) not in sent, "content reached PostHog"


async def test_a_failed_call_reports_its_error(posthog, recorded):
    recorded("anthropic_400_then_openai_400")
    llm = LLM(update_prices=False)
    with pytest.raises(Exception):
        await llm.query(CHAIN, GREET)
    await llm.aclose()
    [event] = posthog
    p = event["properties"]
    assert p["$ai_is_error"] and p["$ai_http_status"] == 400 and "max_output_tokens" in p["$ai_error"]
    assert (p["$ai_provider"], p["retries"], p["fallbacks"]) == ("openai", 2, 1) and "$ai_total_cost_usd" not in p


@pytest.mark.live
async def test_events_reach_posthog(monkeypatch, caplog):
    assert POSTHOG_KEY is not None, "llm_master/.env has no POSTHOG_API_KEY"  # never print the key itself
    monkeypatch.setenv("POSTHOG_API_KEY", POSTHOG_KEY)
    llm = LLM(update_prices=False)
    await llm.query(Model(provider="gemini", model="gemini-3.1-flash-lite"), GREET)
    with caplog.at_level(logging.WARNING, logger="posthog"):
        await llm.aclose()
    assert not [r.getMessage() for r in caplog.records if r.name.startswith("posthog")]
