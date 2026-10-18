"""The facade's held clients, driven against a local server standing in for each provider."""
import gc
import threading
import warnings
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer
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
    """Points every provider at a local server that answers each request with `status`; returns its hit log.

    A None status reads the request and hangs up without answering."""
    def start(status: int | None) -> list[str]:
        hits: list[str] = []

        class Answer(BaseHTTPRequestHandler):
            def do_POST(self):
                hits.append(self.path)
                self.rfile.read(int(self.headers["Content-Length"]))
                if status is None:
                    return
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "refused"}}')

            def log_message(self, format, *args):
                pass

        server = ThreadingTCPServer(("127.0.0.1", 0), Answer)  # HTTPServer would reverse-DNS the host
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        request.addfinalizer(server.server_close)
        request.addfinalizer(server.shutdown)
        for var in BASE_URL_VARS.values():
            monkeypatch.setenv(var, f"http://127.0.0.1:{server.server_address[1]}")
        return hits
    return start


@pytest.fixture
async def llm():
    """A facade closed at teardown, as an app closes its own at shutdown."""
    llm = LLM()
    yield llm
    await llm.aclose()


async def ask(llm, provider):
    await llm.client(provider).acompletion(model="m", messages=[{"role": "user", "content": "hi"}])


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
async def test_sdk_retries_are_off(provider, serve, llm):
    hits = serve(500)  # the openai and anthropic SDKs retry a 500 by default
    with pytest.raises(ProviderError):
        await ask(llm, provider)
    assert len(hits) == 1, hits


@pytest.mark.parametrize("provider", BASE_URL_VARS)
async def test_errors_come_back_typed(provider, serve, llm, monkeypatch):
    monkeypatch.delenv("ANY_LLM_UNIFIED_EXCEPTIONS")  # conftest's global switch would hide a missing flag
    serve(401)
    with pytest.raises(AuthenticationError):
        await ask(llm, provider)


GENAI_RETRIES_DROPS = pytest.mark.xfail(strict=True, raises=AssertionError,
                                        reason="googleapis/python-genai#3160: genai on aiohttp retries a drop by itself")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "fireworks",
                                      pytest.param("gemini", marks=GENAI_RETRIES_DROPS)])
async def test_dropped_connection_is_not_retried(provider, serve, llm):
    hits = serve(None)
    with pytest.raises(ProviderError):
        await ask(llm, provider)
    assert len(hits) == 1, hits


async def test_aclose_closes_every_client_and_the_next_call_reopens(serve):
    hits = serve(500)
    llm = LLM()
    for provider in BASE_URL_VARS:
        with pytest.raises(ProviderError):
            await ask(llm, provider)
    await llm.aclose()
    with pytest.raises(ProviderError):
        await ask(llm, "openai")  # a closed client would never reach the server
    assert len(hits) == len(BASE_URL_VARS) + 1, hits
    await llm.aclose()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del llm
        gc.collect()
    assert not [w for w in caught if "Unclosed" in str(w.message)], caught
