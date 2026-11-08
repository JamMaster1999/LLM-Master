"""SPEC.md's two benchmark gates, live: async_query at 250 in flight, and streaming time to first token.
Each runs the same requests through llm_master and straight through any-llm, on the same aiohttp
transport, so the gap is our code. About 2,500 short requests; a few cents. Each latency runs from the batch's
start to that result's arrival, on both paths.

    uv run --frozen --env-file llm_master/.env python bench/gates.py

Prints markdown tables. Keys already in the environment win over the file's.
"""
import asyncio
import statistics
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field

import aiohttp

from llm_master.facade import LLM
from llm_master.fanout import Request
from llm_master.io.messages import Message, validate
from llm_master.query import build
from llm_master.types import Config, Model

N = 250  # requests per run, all in flight at once
FIRST_TOKENS = 20  # streams per path
LUNA = Model(provider="openai", model="gpt-5.6-luna", config=Config(thinking="none", max_tokens=16))
HAIKU = Model(provider="anthropic", model="claude-haiku-5-5", config=Config(thinking="none", max_tokens=16))
GONE = LUNA.model_copy(update={"model": "gpt-no-such-model"})  # a 404 on every call, so each one fails over

_connectors: list[aiohttp.BaseConnector] = []
_init = aiohttp.BaseConnector.__init__


def _register(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    _connectors.append(self)


aiohttp.BaseConnector.__init__ = _register  # every SDK's pool, so a run can watch its connections


@dataclass
class Run:
    name: str
    started: float = 0.0
    wall: float = 0.0
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failed_tries: int = 0
    peak: int = 0  # connections in use at once, across every pool
    limits: set[int] = field(default_factory=set)

    def row(self) -> str:
        p50 = statistics.median(self.latencies) if self.latencies else float("nan")
        p95 = statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) > 1 else float("nan")
        errors = ", ".join(f"{e} ×{self.errors.count(e)}" for e in sorted(set(self.errors))) or "none"
        limits = "/".join(str(n) for n in sorted(self.limits)) or "?"
        slowest = max(self.latencies, default=float("nan"))
        return (f"| {self.name} | {self.wall:.1f} | {p50:.2f} | {p95:.2f} | {slowest:.2f} | {errors} "
                f"| {self.failed_tries} | {self.peak} (limit {limits}) |")


def history(n: int) -> list[Message]:
    return validate([{"role": "user", "content": f"Reply with the number {n} and nothing else."}])


async def watched(run: Run, work: Awaitable[None]) -> Run:
    """Runs work, sampling the pools every 10 ms."""
    first = len(_connectors)

    async def sample():
        while True:
            pools = [c for c in _connectors[first:] if not c.closed]
            run.peak = max(run.peak, sum(len(c._acquired) for c in pools))  # aiohttp's private in-use set
            run.limits |= {c.limit for c in pools}
            await asyncio.sleep(0.01)

    sampler = asyncio.create_task(sample())
    run.started = time.monotonic()
    try:
        await work
    finally:
        run.wall = time.monotonic() - run.started
        sampler.cancel()
        with suppress(asyncio.CancelledError):
            await sampler  # a sampler that crashed raises here, rather than print a peak of 0
    return run


def facade(per_minute: int | None = None) -> LLM:
    """A fresh LLM, so its pools start cold."""
    return LLM(per_minute=per_minute, update_prices=False)  # no price fetch inside the measured burst


async def llm_master(name: str, candidates: list[Model], *, per_minute: int | None = None) -> Run:
    llm, run = facade(per_minute), Run(name)
    requests: list[Request] = [{"messages": history(n)} for n in range(N)]

    async def work():
        async for _, result in llm.async_query(candidates, requests, concurrency=N):
            if isinstance(result, Exception):
                run.errors.append(type(result).__name__)
            else:
                run.latencies.append(time.monotonic() - run.started)
                run.failed_tries += sum(len(a.errors) for a in result.attempts)
    try:
        return await watched(run, work())
    finally:
        await llm.aclose()


Send = Callable[[LLM, dict, dict, Run], Awaitable[object]]


async def direct(name: str, send: Send) -> Run:
    """The same requests through the facade's own any-llm clients, so the transport is the same. Each request's
    SDK payloads are built before the clock starts: building them is our code, not any-llm's."""
    llm, run = facade(), Run(name)
    payloads = [(luna_request(n), build(HAIKU, history(n), False)[1]) for n in range(N)]

    async def one(luna_payload: dict, haiku_payload: dict):
        try:
            await send(llm, luna_payload, haiku_payload, run)
        except Exception as e:
            run.errors.append(type(e).__name__)
            return
        run.latencies.append(time.monotonic() - run.started)

    async def work():
        await asyncio.gather(*(one(*p) for p in payloads))
    try:
        return await watched(run, work())
    finally:
        await llm.aclose()


def luna_request(n: int, stream: bool = False) -> dict:
    return build(LUNA, history(n), stream)[1]


async def luna(llm: LLM, luna_payload: dict, haiku_payload: dict, run: Run):
    return await llm.client("openai").aresponses(model=LUNA.model, **luna_payload)


async def failover(llm: LLM, luna_payload: dict, haiku_payload: dict, run: Run):
    try:
        return await llm.client("openai").aresponses(model=GONE.model, **luna_payload)
    except Exception:
        run.failed_tries += 1
        return await llm.client("anthropic").amessages(model=HAIKU.model, **haiku_payload)


async def fan_out() -> list[Run]:
    def ours():
        return llm_master("luna answers · llm_master, per_minute=250", [LUNA], per_minute=N)

    def theirs():
        return direct("luna answers · any-llm", luna)
    runs = [await ours(), await theirs(), await theirs(), await ours()]  # ABBA, so drift hits both alike
    runs.append(await llm_master("a missing model fails, Haiku answers · llm_master", [GONE, HAIKU]))
    runs.append(await direct("a missing model fails, Haiku answers · any-llm", failover))
    runs.append(await llm_master("a missing model fails, Haiku answers · llm_master, per_minute=250", [GONE, HAIKU],
                                 per_minute=N))
    return runs


async def first_tokens() -> tuple[list[float], list[float]]:
    """Seconds to the first text, per stream, alternating the paths so drift hits both alike."""
    llm, held = facade(), facade()
    ours, theirs = [], []
    try:
        for n in range(FIRST_TOKENS):
            began, first = time.monotonic(), None
            async for chunk in await llm.query(LUNA, history(n), stream=True):
                if first is None and chunk["type"] == "text-delta":
                    first = time.monotonic() - began
            ours.append(first or float("nan"))
            began, first = time.monotonic(), None
            events = await held.client("openai").aresponses(model=LUNA.model, **luna_request(n, stream=True))
            async for event in events:
                if first is None and event.type == "response.output_text.delta":
                    first = time.monotonic() - began
            theirs.append(first or float("nan"))
    finally:
        await llm.aclose()
        await held.aclose()
    return ours, theirs


async def main():
    print(f"## async_query, {N} requests, all in flight\n")
    print("| run | wall (s) | p50 (s) | p95 (s) | max (s) | errors | failed tries | peak connections |")
    print("|---|---|---|---|---|---|---|---|")
    for run in await fan_out():
        print(run.row())
    ours, theirs = await first_tokens()
    print(f"\n## Streaming time to first token, {FIRST_TOKENS} streams each\n")
    print("| path | p50 (s) | p95 (s) |\n|---|---|---|")
    for name, times in (("llm_master", ours), ("any-llm", theirs)):
        print(f"| {name} | {statistics.median(times):.2f} | {statistics.quantiles(times, n=20)[18]:.2f} |")


if __name__ == "__main__":
    asyncio.run(main())
