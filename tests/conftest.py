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
_ENV = pathlib.Path(__file__).parent.parent / "llm_master" / ".env"
if _ENV.exists():  # absent on fresh checkouts: offline tests need no keys
    for _line in _ENV.read_text().splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep:
            os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))
POSTHOG_KEY = os.environ.pop("POSTHOG_API_KEY", None)  # no call sends telemetry unless its test sets the key


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
    llm = LLM(update_prices=False)  # never fetches prices: the offline tests stay offline
    yield llm
    await llm.aclose()


def local_server(request, answer: Callable[[BaseHTTPRequestHandler, bytes], None]) -> int:
    """A server on localhost for one test: answer(handler, body) writes each POST's reply. Returns its port."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            answer(self, self.rfile.read(int(self.headers["Content-Length"])))

        def log_message(self, format, *args):
            pass

    server = ThreadingTCPServer(("127.0.0.1", 0), Handler)  # HTTPServer would reverse-DNS the host
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    request.addfinalizer(server.server_close)
    request.addfinalizer(server.shutdown)
    return server.server_address[1]


@pytest.fixture
def provider_server(monkeypatch, request):
    """Points every provider at a local server; answer(handler, provider, path, body) writes each reply.
    The keys are fake unless RECORD=1, which forwards each request to the real provider."""
    def start(answer: Callable[[BaseHTTPRequestHandler, str, str, bytes], None]) -> None:
        def route(handler: BaseHTTPRequestHandler, body: bytes) -> None:
            provider, _, rest = handler.path[1:].partition("/")
            answer(handler, provider, "/" + rest, body)

        port = local_server(request, route)
        for provider, (var, _) in UPSTREAM.items():
            monkeypatch.setenv(var, f"http://127.0.0.1:{port}/{provider}")
        for key in KEYS if os.environ.get("RECORD") != "1" else ():
            monkeypatch.setenv(key, "test")
    return start


def exchanges(name: str) -> list[dict]:
    """A recording's request and response pairs, in order."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


def failing(name: str, status: int, error: dict, *, times: int = 1, headers: dict | None = None) -> list[dict]:
    """name's first request, answered `times` times with a crafted error: providers don't fail on cue."""
    reply = {"status": status, "type": "application/json", "body": json.dumps(error), "headers": headers or {}}
    return [{"request": exchanges(name)[0]["request"], "response": reply}] * times


def cut(name: str, before: str, tail: str = "") -> list[dict]:
    """name's first exchange, its stream ended before the first event holding `before`, then `tail`."""
    first = exchanges(name)[0]
    body = first["response"]["body"]
    end = "\r\n\r\n" if "\r\n\r\n" in body else "\n\n"  # openai's speech stream ends events with CRLFs
    events = body.split(end)
    kept = events[:next(i for i, event in enumerate(events) if before in event)]
    return [{**first, "response": {**first["response"], "body": "".join(e + end for e in kept) + tail}}]


@pytest.fixture
def recorded(provider_server, request):
    """Stands in for every provider, answering from the steps in order, and checks at teardown that each
    request body matches the recorded one: the goldens of the same-bytes rule. A step is a fixture name,
    fixtures/<name>.json, or a list of exchanges built by failing() or cut(), which replay as given.
    RECORD=1 forwards a named step's requests to the real provider and rewrites its file (bodies only,
    never keys). A multipart body (an image edit) is pinned as base64 with its random boundary blanked."""
    def serve(*steps: str | list[dict]) -> None:
        live = os.environ.get("RECORD") == "1"
        new = live and len(steps) == 1 and isinstance(steps[0], str)  # one name, recorded afresh
        plan = [] if new else [(step if isinstance(step, str) else None, e)
                               for step in steps for e in (exchanges(step) if isinstance(step, str) else step)]
        received: list[dict] = []
        recordings: dict[str, list[dict]] = {}

        def answer(handler: BaseHTTPRequestHandler, provider: str, path: str, body: bytes) -> None:
            kind = handler.headers.get("Content-Type", "")
            if kind.startswith("multipart/"):
                pinned = base64.b64encode(body.replace(kind.split("boundary=")[1].encode(), b"BOUNDARY")).decode()
            else:
                pinned = json.loads(body)
            received.append({"path": path, "body": pinned})
            name, exchange = (steps[0], None) if new else plan[len(received) - 1]
            if live and isinstance(name, str):
                headers = {k: v for k, v in handler.headers.items()
                           if k.lower() not in ("host", "content-length", "accept-encoding")}
                upstream = httpx.post(f"{UPSTREAM[provider][1]}{path}", content=body, headers=headers, timeout=600)
                kind = upstream.headers.get("content-type", "")
                text = kind.startswith(("application/json", "text/"))  # else binary, kept as base64
                reply = {"status": upstream.status_code, "type": kind, **(
                    {"body": upstream.text} if text else {"base64": base64.b64encode(upstream.content).decode()})}
                recordings.setdefault(name, []).append({"request": received[-1], "response": reply})
            else:
                assert exchange is not None
                reply = exchange["response"]
            payload = reply["body"].encode() if "body" in reply else base64.b64decode(reply["base64"])
            handler.send_response(reply["status"])
            for header, value in reply.get("headers", {}).items():  # a crafted Retry-After, say
                handler.send_header(header, value)
            handler.send_header("Content-Type", reply["type"])
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        provider_server(answer)

        def check():
            for name, recording in recordings.items():
                (FIXTURES / f"{name}.json").write_text(json.dumps(recording, indent=1) + "\n")
            if not live:
                names = " + ".join(step if isinstance(step, str) else "crafted" for step in steps)
                assert received == [e["request"] for _, e in plan], f"{names}: the request changed"
        request.addfinalizer(check)
    return serve
