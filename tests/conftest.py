"""Live-test env: keys from the repo .env, typed any-llm errors on. And the recorded provider stand-in."""
import base64
import json
import os
import pathlib
import secrets
import threading
from collections.abc import Callable
from itertools import takewhile
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import httpx
import pytest

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
os.environ.setdefault("MODAL_ENVIRONMENT", "dev")  # sandboxes and apps never run beside production's
# server/app.py reads these at import: each worker's own app, public while the tests run, with a short
# prompt and the fixture skills
os.environ.update(AGENT_APP=f"llm-master-test-{secrets.token_hex(4)}", AGENT_PUBLIC="1",
                  AGENT_PROMPT=str(pathlib.Path(__file__).parent / "fixtures" / "prompt.md"),
                  AGENT_SKILLS=str(pathlib.Path(__file__).parent / "tools" / "skills"))
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
            "fireworks": ("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference")}
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "FIREWORKS_API_KEY")


def pytest_xdist_auto_num_workers(config):
    """Live tests mostly wait on the network, so they get more workers than cores."""
    return os.cpu_count() if config.option.markexpr == "not live" else 32


def pytest_collection_modifyitems(config, items):
    """Agentic tests only when -m names them. Live tests first, so no parallel worker starts a slow
    one last."""
    if "agentic" not in config.option.markexpr:
        if skipped := [item for item in items if item.get_closest_marker("agentic")]:
            config.hook.pytest_deselected(items=skipped)
            items[:] = [item for item in items if not item.get_closest_marker("agentic")]
    items.sort(key=lambda item: item.get_closest_marker("live") is None)


@pytest.fixture
async def llm():
    from llm_master.facade import LLM
    llm = LLM(update_prices=False)  # never fetches prices: the offline tests stay offline
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
            # any-llm wants fireworks' base to end in /v1, and the fireworks SDK puts it back on each path
            v1 = "/v1" if provider == "fireworks" else ""
            monkeypatch.setenv(var, f"http://127.0.0.1:{server.server_address[1]}/{provider}{v1}")
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
    kept = takewhile(lambda event: before.encode() not in event, _events(first["response"]["body"].encode()))
    return [{**first, "response": {**first["response"], "body": b"".join(kept).decode() + tail}}]


def paced(steps: list[dict], *pauses: float) -> list[dict]:
    """steps, each reply sent an event at a time after its pause in seconds, the last pause repeating: (0.3,) is a
    slow stream, (0, 60) one event and then quiet. The stand-in stops waiting at teardown."""
    return [{**e, "response": {**e["response"], "pauses": pauses}} for e in steps]


def _events(payload: bytes) -> list[bytes]:
    """payload in its stream's events, ends kept: a JSON body is one."""
    end = b"\r\n\r\n" if b"\r\n\r\n" in payload else b"\n\n"  # openai's speech stream ends events with CRLFs
    *events, rest = payload.split(end)
    return [e + end for e in events] + ([rest] if rest else [])


@pytest.fixture
def recorded(provider_server, request):
    """Stands in for every provider, answering from the steps in order, and checks at teardown that each
    request body matches the recorded one: the goldens of the same-bytes rule. A step is a fixture name,
    fixtures/<name>.json, or a list of exchanges built by failing(), cut() or paced(), which replay as given.
    RECORD=1 with one name forwards its requests to the real provider and rewrites its file (bodies only,
    never keys). A multipart body (an image edit) is pinned as base64 with its random boundary blanked."""
    def serve(*steps: str | list[dict]) -> None:
        live = os.environ.get("RECORD") == "1" and len(steps) == 1 and isinstance(steps[0], str)
        plan = [] if live else [e for step in steps for e in (exchanges(step) if isinstance(step, str) else step)]
        received: list[dict] = []
        torn_down = threading.Event()

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
                upstream = httpx.post(f"{UPSTREAM[provider][1]}{path}", content=body, headers=headers, timeout=600)
                kind = upstream.headers.get("content-type", "")
                text = kind.startswith(("application/json", "text/"))  # else binary, kept as base64
                reply = {"status": upstream.status_code, "type": kind, **(
                    {"body": upstream.text} if text else {"base64": base64.b64encode(upstream.content).decode()})}
                plan.append({"request": received[-1], "response": reply})
            else:
                reply = plan[len(received) - 1]["response"]
            payload = reply["body"].encode() if "body" in reply else base64.b64decode(reply["base64"])
            handler.send_response(reply["status"])
            for header, value in reply.get("headers", {}).items():  # a crafted Retry-After, say
                handler.send_header(header, value)
            handler.send_header("Content-Type", reply["type"])
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            pauses = reply.get("pauses", [0])
            for i, event in enumerate(_events(payload)):
                if torn_down.wait(pauses[min(i, len(pauses) - 1)]):
                    return
                handler.wfile.write(event)

        provider_server(answer)
        request.addfinalizer(torn_down.set)  # ahead of the server's close, which waits for every reply

        def check():
            if live:
                (FIXTURES / f"{steps[0]}.json").write_text(json.dumps(plan, indent=1) + "\n")
            else:
                names = " + ".join(step if isinstance(step, str) else "crafted" for step in steps)
                assert received == [e["request"] for e in plan], f"{names}: the request changed"
        request.addfinalizer(check)
    return serve
