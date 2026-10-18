"""Live-test env: keys from the repo .env, typed any-llm errors on. And the recorded provider stand-in."""
import json
import os
import pathlib
import threading
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
def recorded(monkeypatch, request):
    """Stands in for every provider, answering from fixtures/<name>.json in order, and checks at teardown
    that each request body matches the recorded one: the goldens of the same-bytes rule.
    RECORD=1 forwards to the real provider instead and rewrites the file (bodies only, never keys)."""
    def serve(name: str) -> None:
        path, live = FIXTURES / f"{name}.json", os.environ.get("RECORD") == "1"
        exchanges = [] if live else json.loads(path.read_text())
        received: list[dict] = []

        class Answer(BaseHTTPRequestHandler):
            def do_POST(self):
                provider, _, rest = self.path[1:].partition("/")
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append({"path": "/" + rest, "body": json.loads(body)})
                if live:
                    headers = {k: v for k, v in self.headers.items()
                               if k.lower() not in ("host", "content-length", "accept-encoding")}
                    reply = httpx.post(f"{UPSTREAM[provider][1]}/{rest}", content=body, headers=headers, timeout=600)
                    exchanges.append({"request": received[-1], "response": {
                        "status": reply.status_code, "type": reply.headers.get("content-type", ""), "body": reply.text}})
                answer = exchanges[len(received) - 1]["response"]
                self.send_response(answer["status"])
                self.send_header("Content-Type", answer["type"])
                self.send_header("Content-Length", str(len(answer["body"].encode())))
                self.end_headers()
                self.wfile.write(answer["body"].encode())

            def log_message(self, format, *args):
                pass

        server = ThreadingTCPServer(("127.0.0.1", 0), Answer)  # HTTPServer would reverse-DNS the host
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        for provider, (var, _) in UPSTREAM.items():
            monkeypatch.setenv(var, f"http://127.0.0.1:{server.server_address[1]}/{provider}")
        for key in KEYS:
            if not live:
                monkeypatch.setenv(key, "test")

        def check():
            server.shutdown()
            server.server_close()
            if live:
                path.write_text(json.dumps(exchanges, indent=1) + "\n")
            else:
                assert received == [e["request"] for e in exchanges], f"{name}: the request changed"
        request.addfinalizer(check)
    return serve
