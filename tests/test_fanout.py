"""async_query(): many query() calls at once, against a stand-in Anthropic that echoes each prompt."""
import asyncio
import json
import socket
import threading
import time
from contextlib import aclosing

import pytest
from any_llm.exceptions import InvalidRequestError

from llm_master.io.response import Response
from llm_master.types import Model

HAIKU = Model(provider="anthropic", model="claude-haiku-5-5")


class Echo:
    """What the stand-in saw: the most requests in flight at once, and the slow requests' arrivals and
    hang-ups. until(test) waits, in a thread, for test(seen) to hold."""

    def __init__(self) -> None:
        self.seen = {"now": 0, "most": 0, "slow": 0, "hung up": 0}
        self.changed = threading.Condition()

    def count(self, key: str, by: int = 1) -> None:
        with self.changed:
            self.seen[key] += by
            self.seen["most"] = max(self.seen["most"], self.seen["now"])
            self.changed.notify_all()

    def until(self, test, timeout: float = 5) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: test(self.seen), timeout)


@pytest.fixture
def echo(provider_server):
    """Answers each request with its own prompt. Ordinary prompts pass in pairs, so two are in flight
    together; "bad" gets a 400; "slow" never answers and notes when its client hangs up; "first"
    answers once two slow requests are in."""
    echo, pair = Echo(), threading.Barrier(2, timeout=5)

    def answer(handler, provider, path, body):
        prompt = json.loads(body)["messages"][-1]["content"]
        status, reply = 200, {"id": "msg", "type": "message", "role": "assistant", "model": HAIKU.model,
                              "content": [{"type": "text", "text": prompt}], "stop_reason": "end_turn",
                              "usage": {"input_tokens": 1, "output_tokens": 1}}
        if prompt == "slow":
            echo.count("slow")
            handler.connection.settimeout(10)
            if handler.connection.recv(1, socket.MSG_PEEK) == b"":  # the request sent, only a hang-up is left
                echo.count("hung up")
            return
        if prompt == "first":
            echo.until(lambda seen: seen["slow"] == 2)
        elif prompt == "bad":
            status, reply = 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
        else:
            echo.count("now")
            pair.wait()
            time.sleep(0.1)  # held, so a request past the cap would arrive while two are in
            echo.count("now", -1)
        data = json.dumps(reply).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    provider_server(answer)
    return echo


def ask(*prompts: str) -> list:
    return [{"messages": [{"role": "user", "content": p}]} for p in prompts]


def answers(results: dict) -> dict:
    return {i: r.message.content for i, r in results.items() if isinstance(r, Response)}


async def test_at_most_concurrency_run_at_once_and_each_result_carries_its_index(echo, llm):
    results = {i: r async for i, r in llm.async_query(HAIKU, ask(*"012345"), concurrency=2)}
    assert answers(results) == {i: str(i) for i in range(6)}
    assert echo.seen["most"] == 2


async def test_a_failing_request_leaves_the_rest_to_finish(echo, llm):
    results = {i: r async for i, r in llm.async_query(HAIKU, ask("a", "bad", "b"))}
    assert isinstance(results[1], InvalidRequestError)
    assert answers(results) == {0: "a", 2: "b"}


async def test_closing_early_cancels_the_requests_still_running(echo, llm):
    async with aclosing(llm.async_query(HAIKU, ask("slow", "first", "slow"))) as results:
        async for index, _ in results:
            assert index == 1
            break
    assert await asyncio.to_thread(echo.until, lambda seen: seen["hung up"] == 2), "a slow request ran on"


@pytest.mark.parametrize("candidates, requests, concurrency", [
    (HAIKU, [*ask("hi"), {**ask("hi")[0], "kwargs": {"temperature": 0}}], 250),  # v1's config key
    (HAIKU, [*ask("hi"), {"moderation": True}], 250),
    ([], ask("hi"), 250),
    (HAIKU, ask("hi"), 0),
], ids=["v1-key", "no-messages", "no-candidate", "zero-concurrency"])
async def test_a_malformed_batch_sends_nothing(candidates, requests, concurrency, recorded, llm):
    recorded("nothing_sent")
    with pytest.raises(ValueError):
        async for _ in llm.async_query(candidates, requests, concurrency=concurrency):
            pass


@pytest.mark.live
async def test_each_request_falls_back_and_answers_at_its_index_live(llm):
    candidates = [Model(provider="openai", model="gpt-no-such-model"), Model(provider="gemini", model="gemini-3.1-flash-lite")]
    requests = ask(*(f"Reply with the number {n} and nothing else." for n in "123"))
    results = {i: r async for i, r in llm.async_query(candidates, requests)}
    assert {i: a.strip() for i, a in answers(results).items()} == {0: "1", 1: "2", 2: "3"}
    assert all(isinstance(r, Response) and len(r.attempts) == 2 for r in results.values())
