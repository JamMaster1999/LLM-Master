"""The facade's held clients, driven against a local server standing in for each provider."""
import gc
import random
import warnings
from typing import get_args

import pytest
from any_llm.exceptions import AuthenticationError, ProviderError
from conftest import KEYS, UPSTREAM

from llm_master.facade import LLM
from llm_master.providers import ProviderName

GENAI_RETRIES_DROPS = pytest.mark.xfail(strict=True, raises=AssertionError,
                                        reason="googleapis/python-genai#3160: genai on aiohttp retries a drop by itself")


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    for var in (*KEYS, "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.setenv(var, "test")


@pytest.fixture
def serve(provider_server):
    """Every provider answers `status` (None hangs up); returns the hit log."""
    def start(status: int | None) -> list[str]:
        hits: list[str] = []

        def answer(handler, provider, path, body):
            hits.append(path)
            if status:
                handler.send_response(status)
                handler.end_headers()
        provider_server(answer)
        return hits
    return start


async def ask(llm, provider):
    await llm.client(provider).acompletion(model="m", messages=[{"role": "user", "content": "hi"}])


def test_one_held_client_per_provider():
    llm = LLM()
    roster = get_args(ProviderName)
    assert [llm.client(p).PROVIDER_NAME for p in roster] == list(roster)
    assert llm.client("openai") is llm.client("openai")


def test_unknown_provider_raises():
    with pytest.raises(KeyError):  # any-llm serves mistral; the roster does not
        LLM().client("mistral")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("provider", UPSTREAM)
async def test_sdk_retries_are_off(provider, serve, llm):
    hits = serve(500)  # the openai and anthropic SDKs retry a 500 by default
    with pytest.raises(ProviderError):
        await ask(llm, provider)
    assert len(hits) == 1, hits


@pytest.mark.parametrize("provider", ["openai", "anthropic", "fireworks",
                                      pytest.param("gemini", marks=GENAI_RETRIES_DROPS)])
async def test_dropped_connection_is_not_retried(provider, serve, llm, monkeypatch):
    monkeypatch.setattr(random, "randint", lambda a, b: a)  # genai waits 1 + randint(0, 9) s before its retry
    hits = serve(None)
    with pytest.raises(ProviderError):
        await ask(llm, provider)
    assert len(hits) == 1, hits


@pytest.mark.parametrize("provider", UPSTREAM)
async def test_errors_come_back_typed(provider, serve, llm, monkeypatch):
    monkeypatch.delenv("ANY_LLM_UNIFIED_EXCEPTIONS")  # conftest's global switch would hide a missing flag
    serve(401)
    with pytest.raises(AuthenticationError):
        await ask(llm, provider)


async def test_aclose_closes_and_forgets_every_client(serve):
    hits = serve(500)
    llm = LLM()
    for provider in UPSTREAM:
        with pytest.raises(ProviderError):
            await ask(llm, provider)
    await llm.aclose()
    with pytest.raises(ProviderError):
        await ask(llm, "openai")  # a closed client would never reach the server
    assert len(hits) == len(UPSTREAM) + 1, hits
    await llm.aclose()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del llm
        gc.collect()
    assert not [w for w in caught if "Unclosed" in str(w.message)], caught
