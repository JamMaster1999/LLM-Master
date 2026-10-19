"""async_query(): many query() calls at once, against a stand-in Anthropic that echoes each prompt."""
import asyncio
import json
import threading

import pytest
from any_llm.exceptions import InvalidRequestError

from llm_master.types import Model

HAIKU = Model(provider="anthropic", model="claude-haiku-5-5")


@pytest.fixture
def echo(provider_server, request):
    """Answers each request with its own prompt, and counts the requests in flight. Ordinary prompts
    pass in pairs, so two are in flight together; "slow" waits for teardown; "bad" gets a 400 once
    both slow requests are in."""
    seen = {"now": 0, "most": 0, "slow": 0}
    lock = threading.Lock()
    arrived, pair, release = threading.Condition(lock), threading.Barrier(2, timeout=5), threading.Event()
    request.addfinalizer(release.set)

    def answer(handler, provider, path, body):
        prompt = json.loads(body)["messages"][-1]["content"]
        status, reply = 200, {"id": "msg", "type": "message", "role": "assistant", "model": HAIKU.model,
                              "content": [{"type": "text", "text": prompt}], "stop_reason": "end_turn",
                              "usage": {"input_tokens": 1, "output_tokens": 1}}
        if prompt == "slow":
            with arrived:
                seen["slow"] += 1
                arrived.notify_all()
            release.wait(10)
        elif prompt == "bad":
            with arrived:
                arrived.wait_for(lambda: seen["slow"] == 2, timeout=5)
            status, reply = 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
        else:
            with lock:
                seen["now"] += 1
                seen["most"] = max(seen["most"], seen["now"])
            pair.wait()
            with lock:
                seen["now"] -= 1
        data = json.dumps(reply).encode()
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        except OSError:  # a cancelled request's client hung up
            pass
    provider_server(answer)
    return seen


def ask(*prompts: str) -> list:
    return [{"messages": [{"role": "user", "content": p}]} for p in prompts]


async def test_at_most_concurrency_run_at_once_and_answers_keep_request_order(echo, llm):
    progress = []
    responses = await llm.async_query(HAIKU, ask(*"012345"), concurrency=2,
                                      progress_callback=lambda done, total: progress.append((done, total)))
    assert [r.message.content for r in responses] == list("012345")
    assert echo["most"] == 2 and progress == [(n, 6) for n in range(1, 7)]


async def test_a_failing_request_cancels_the_rest_and_raises_its_error(echo, llm):
    with pytest.raises(InvalidRequestError):  # had it waited for the slow requests, wait_for would time out
        await asyncio.wait_for(llm.async_query(HAIKU, ask("slow", "bad", "slow")), 5)
    assert echo["slow"] == 2, "the slow requests weren't in flight when bad failed"


async def test_a_v1_request_key_sends_nothing(recorded, llm):
    recorded("nothing_sent")
    with pytest.raises(ValueError, match="kwargs"):
        await llm.async_query(HAIKU, [*ask("hi"), {"messages": [{"role": "user", "content": "hi"}],
                                                    "kwargs": {"temperature": 0}}])  # pyright: ignore[reportArgumentType]


@pytest.mark.live
async def test_answers_keep_request_order_live(llm):
    candidate = Model(provider="gemini", model="gemini-3.1-flash-lite")
    responses = await llm.async_query(candidate, ask(*(f"Reply with the number {n} and nothing else." for n in "123")))
    assert [r.message.content.strip() for r in responses] == ["1", "2", "3"]
