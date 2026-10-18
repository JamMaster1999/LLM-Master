"""Live-test env: keys from the repo .env, typed any-llm errors on. And the recorded provider stand-in."""
import json
import os
import pathlib
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import httpx
import pytest

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
_ENV = pathlib.Path(__file__).parent.parent / "llm_master" / ".env"
if _ENV.exists():  # absent on fresh checkouts: offline tests need no keys
    for _line in _ENV.read_text().splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep:
            os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))


FIXTURES = pathlib.Path(__file__).parent / "fixtures"
# vertexai is absent: each of its requests first fetches a real GCP login token
UPSTREAM = {"openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "anthropic": ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            "gemini": ("GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
            "fireworks": ("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1")}
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "FIREWORKS_API_KEY")


@pytest.fixture
async def llm():
    from llm_master.facade import LLM
    llm = LLM()
    yield llm
    await llm.aclose()


@pytest.fixture
def provider_server(monkeypatch, request):
    """Points every provider at a local server; answer(handler, provider, path, body) writes each reply."""
    def start(answer: Callable[[BaseHTTPRequestHandler, str, str, bytes], None]) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                provider, _, rest = self.path[1:].partition("/")
                answer(self, provider, "/" + rest, self.rfile.read(int(self.headers["Content-Length"])))

            def log_message(self, format, *args):
                pass

        server = ThreadingTCPServer(("127.0.0.1", 0), Handler)  # HTTPServer would reverse-DNS the host
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        request.addfinalizer(server.server_close)
        request.addfinalizer(server.shutdown)
        for provider, (var, _) in UPSTREAM.items():
            monkeypatch.setenv(var, f"http://127.0.0.1:{server.server_address[1]}/{provider}")
    return start


@pytest.fixture
def recorded(provider_server, monkeypatch, request):
    """Stands in for every provider, answering from fixtures/<name>.json in order, and checks at teardown
    that each request body matches the recorded one: the goldens of the same-bytes rule.
    RECORD=1 forwards to the real provider instead and rewrites the file (bodies only, never keys)."""
    def serve(name: str) -> None:
        path, live = FIXTURES / f"{name}.json", os.environ.get("RECORD") == "1"
        exchanges = [] if live else json.loads(path.read_text())
        received: list[dict] = []

        def answer(handler: BaseHTTPRequestHandler, provider: str, path: str, body: bytes) -> None:
            received.append({"path": path, "body": json.loads(body)})
            if live:
                headers = {k: v for k, v in handler.headers.items()
                           if k.lower() not in ("host", "content-length", "accept-encoding")}
                reply = httpx.post(f"{UPSTREAM[provider][1]}{path}", content=body, headers=headers, timeout=600)
                exchanges.append({"request": received[-1], "response": {
                    "status": reply.status_code, "type": reply.headers.get("content-type", ""), "body": reply.text}})
            reply = exchanges[len(received) - 1]["response"]
            handler.send_response(reply["status"])
            handler.send_header("Content-Type", reply["type"])
            handler.send_header("Content-Length", str(len(reply["body"].encode())))
            handler.end_headers()
            handler.wfile.write(reply["body"].encode())

        provider_server(answer)
        for key in KEYS:
            if not live:
                monkeypatch.setenv(key, "test")

        def check():
            if live:
                path.write_text(json.dumps(exchanges, indent=1) + "\n")
            else:
                assert received == [e["request"] for e in exchanges], f"{name}: the request changed"
        request.addfinalizer(check)
    return serve
