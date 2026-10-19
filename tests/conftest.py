"""Live-test env: keys from the repo .env, typed any-llm errors on. And the recorded provider stand-in."""
import base64
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
os.environ.setdefault("MODAL_ENVIRONMENT", "dev")  # sandboxes and apps never run beside production's
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


def pytest_xdist_auto_num_workers(config):
    """Live tests mostly wait on the network, so they get more workers than cores."""
    return os.cpu_count() if config.option.markexpr == "not live" else 32


def pytest_collection_modifyitems(items):
    """Live tests first, so no parallel worker starts a slow one last."""
    items.sort(key=lambda item: item.get_closest_marker("live") is None)


@pytest.fixture
async def llm():
    from llm_master.facade import LLM
    llm = LLM()
    yield llm
    await llm.aclose()


@pytest.fixture
def provider_server(monkeypatch, request):
    """Points every provider at a local server; answer(handler, provider, path, body) writes each reply.
    The keys are fake unless RECORD=1, which forwards each request to the real provider."""
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
        for key in KEYS if os.environ.get("RECORD") != "1" else ():
            monkeypatch.setenv(key, "test")
    return start


@pytest.fixture
def recorded(provider_server, request):
    """Stands in for every provider, answering from fixtures/<name>.json in order, and checks at teardown
    that each request body matches the recorded one: the goldens of the same-bytes rule.
    RECORD=1 forwards to the real provider instead and rewrites the file (bodies only, never keys).
    A multipart body (an image edit) is pinned as base64 with its random boundary blanked."""
    def serve(name: str) -> None:
        path, live = FIXTURES / f"{name}.json", os.environ.get("RECORD") == "1"
        exchanges = [] if live else json.loads(path.read_text())
        received: list[dict] = []

        def answer(handler: BaseHTTPRequestHandler, provider: str, path: str, body: bytes) -> None:
            kind = handler.headers.get("Content-Type", "")
            if kind.startswith("multipart/"):
                pinned = base64.b64encode(body.replace(kind.split("boundary=")[1].encode(), b"BOUNDARY")).decode()
            else:
                pinned = json.loads(body)
            received.append({"path": path, "body": pinned})
            if live:
                headers = {k: v for k, v in handler.headers.items()
                           if k.lower() not in ("host", "content-length", "accept-encoding")}
                reply = httpx.post(f"{UPSTREAM[provider][1]}{path}", content=body, headers=headers, timeout=600)
                kind = reply.headers.get("content-type", "")
                text = kind.startswith(("application/json", "text/"))  # else binary, kept as base64
                exchanges.append({"request": received[-1], "response": {
                    "status": reply.status_code, "type": kind,
                    **({"body": reply.text} if text else {"base64": base64.b64encode(reply.content).decode()})}})
            reply = exchanges[len(received) - 1]["response"]
            payload = reply["body"].encode() if "body" in reply else base64.b64decode(reply["base64"])
            handler.send_response(reply["status"])
            handler.send_header("Content-Type", reply["type"])
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        provider_server(answer)

        def check():
            if live:
                path.write_text(json.dumps(exchanges, indent=1) + "\n")
            else:
                assert received == [e["request"] for e in exchanges], f"{name}: the request changed"
        request.addfinalizer(check)
    return serve
