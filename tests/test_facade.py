"""The facade's held clients, driven against a local server standing in for each provider."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import get_args

import pytest
from any_llm.exceptions import AuthenticationError, ProviderError

from llm_master.facade import LLM
from llm_master.providers import ProviderName

# vertexai is absent: each of its requests first fetches a real GCP login token
BASE_URL_VARS = {"openai": "OPENAI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL",
                 "gemini": "GOOGLE_GEMINI_BASE_URL", "fireworks": "FIREWORKS_API_BASE"}


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    """Fake credentials for every provider: these tests need no keys and never send a real one."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.setenv(var, "test-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")


@pytest.fixture
def serve(monkeypatch, request):
    """Points every provider at a local server that answers each request with `status`; returns its hit log."""
    def start(status: int) -> list[str]:
        hits: list[str] = []

        class Answer(BaseHTTPRequestHandler):
            def do_POST(self):
                hits.append(self.path)
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "refused"}}')

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Answer)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        request.addfinalizer(server.server_close)
        request.addfinalizer(server.shutdown)
        for var in BASE_URL_VARS.values():
            monkeypatch.setenv(var, f"http://127.0.0.1:{server.server_port}")
        return hits
    return start


async def ask(provider):
    await LLM().client(provider).acompletion(model="m", messages=[{"role": "user", "content": "hi"}])


def test_same_provider_reuses_its_client():
    llm = LLM()
    assert llm.client("openai") is llm.client("openai")


def test_each_provider_gets_its_own_client():
    llm = LLM()
    roster = get_args(ProviderName)
    assert [llm.client(p).PROVIDER_NAME for p in roster] == list(roster)


def test_unknown_provider_raises():
    with pytest.raises(KeyError):  # any-llm serves mistral; the roster does not
        LLM().client("mistral")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("provider", BASE_URL_VARS)
async def test_sdk_retries_are_off(provider, serve):
    hits = serve(500)  # the openai and anthropic SDKs retry a 500 by default
    with pytest.raises(ProviderError):
        await ask(provider)
    assert len(hits) == 1, hits


@pytest.mark.parametrize("provider", BASE_URL_VARS)
async def test_errors_come_back_typed(provider, serve, monkeypatch):
    monkeypatch.delenv("ANY_LLM_UNIFIED_EXCEPTIONS")  # conftest's global switch would hide a missing flag
    serve(401)
    with pytest.raises(AuthenticationError):
        await ask(provider)
