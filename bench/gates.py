"""SPEC.md's benchmark gates, live: async_query at 250 in flight against each provider's own SDK, and streaming
time to first token. Three workloads per provider: text, text with an image, and a follow-up to a saved turn.
Each sends the same request bodies through llm_master and straight through the SDK on the same aiohttp
transport, so the gap is ours: any-llm and llm_master together. Text also runs through any-llm alone, which
splits that gap in two. Each latency runs from the batch's start to that result's arrival. About 9,000 short
requests on the cheapest models; well under a dollar.

    uv run --frozen --env-file llm_master/.env python bench/gates.py

Prints markdown tables. Keys already in the environment win over the file's.
"""
import asyncio
import base64
import io
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import anthropic
import openai
from PIL import Image as Pillow

from llm_master.facade import LLM
from llm_master.io.messages import Message, validate
from llm_master.io.parts import Image
from llm_master.query import build
from llm_master.types import Config, Model

N = 250  # requests per run, all in flight at once
FIRST_TOKENS = 20  # streams per path
NANO = Model(provider="openai", model="gpt-5.4-nano", config=Config(thinking="none", max_tokens=16))
HAIKU = Model(provider="anthropic", model="claude-haiku-5-5", config=Config(thinking="none", max_tokens=16))
GONE = NANO.model_copy(update={"model": "gpt-no-such-model"})  # a 404 on every call, so each one fails over

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

    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else float("nan")

    def row(self) -> str:
        p95 = statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) > 1 else float("nan")
        errors = ", ".join(f"{e} ×{self.errors.count(e)}" for e in sorted(set(self.errors))) or "none"
        limits = "/".join(str(n) for n in sorted(self.limits)) or "?"
        slowest = max(self.latencies, default=float("nan"))
        return (f"| {self.name} | {self.wall:.2f} | {self.p50():.2f} | {p95:.2f} | {slowest:.2f} | {errors} "
                f"| {self.failed_tries} | {self.peak} (limit {limits}) |")


def square() -> Image:
    """A small red PNG, the same in every image request."""
    out = io.BytesIO()
    Pillow.new("RGB", (64, 64), "red").save(out, "PNG")
    return Image("data:image/png;base64," + base64.b64encode(out.getvalue()).decode())


SQUARE = square()
Workload = Callable[[int], list[Message]]  # request n's history


def text(n: int) -> list[Message]:
    return validate([{"role": "user", "content": f"Reply with the number {n} and nothing else."}])


def image(n: int) -> list[Message]:
    return validate([{"role": "user", "content": [
        {"type": "text", "text": f"Reply with the number {n}, then the square's colour, and nothing else."}, SQUARE]}])


FIRST = {"role": "user", "content": "Pick a colour and reply with only its name."}


async def follow_up(llm: LLM, model: Model) -> Workload:
    """A follow-up to one saved turn: the first question, the model's real answer as our history keeps it,
    then request n's question. The turn is made once, before the clock."""
    answered = await llm.query(model, [FIRST])

    def history(n: int) -> list[Message]:
        return validate([FIRST, answered.message,
                         {"role": "user", "content": f"Now reply with the number {n} and nothing else."}])
    return history


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


async def llm_master(name: str, candidates: Sequence[Model], histories: list[list[Message]], *,
                     per_minute: int | None = None) -> Run:
    llm, run = facade(per_minute), Run(name)

    async def work():
        async for _, result in llm.async_query(candidates, histories, concurrency=N):
            if isinstance(result, Exception):
                run.errors.append(type(result).__name__)
            else:
                run.latencies.append(time.monotonic() - run.started)
                run.failed_tries += sum(len(a.errors) for a in result.attempts)
    try:
        return await watched(run, work())
    finally:
        await llm.aclose()


def body(model: Model, history: list[Message], stream: bool = False) -> dict:
    """The request llm_master sends for history, as the provider's SDK takes it: any-llm calls responses'
    input input_data."""
    kwargs = build(model, history, stream)[1]
    if model.provider == "openai":
        kwargs["input"] = kwargs.pop("input_data")
    return kwargs


class Native:
    """The providers' own SDKs on their aiohttp transports, retries off: the baseline."""

    def __init__(self) -> None:
        self.openai = openai.AsyncOpenAI(http_client=openai.DefaultAioHttpClient(), max_retries=0)
        self.anthropic = anthropic.AsyncAnthropic(http_client=anthropic.DefaultAioHttpClient(), max_retries=0)

    async def send(self, model: Model, kwargs: dict) -> Any:
        if model.provider == "openai":
            return await self.openai.responses.create(model=model.model, **kwargs)
        return await self.anthropic.messages.create(model=model.model, **kwargs)

    async def aclose(self) -> None:
        await self.openai.close()
        await self.anthropic.close()


async def baseline(name: str, sends: list[Callable[[], Awaitable[object]]], close: Callable[[], Awaitable[None]],
                   run: Run | None = None) -> Run:
    """Every send at once, the bodies built before the clock: building them is our code."""
    run = run or Run(name)

    async def one(send: Callable[[], Awaitable[object]]):
        try:
            await send()
        except Exception as e:
            run.errors.append(type(e).__name__)
            return
        run.latencies.append(time.monotonic() - run.started)

    async def work():
        await asyncio.gather(*(one(send) for send in sends))
    try:
        return await watched(run, work())
    finally:
        await close()


async def native(name: str, model: Model, histories: list[list[Message]]) -> Run:
    sdk = Native()
    bodies = [body(model, h) for h in histories]
    return await baseline(name, [lambda b=b: sdk.send(model, b) for b in bodies], sdk.aclose)


async def any_llm(name: str, model: Model, histories: list[list[Message]]) -> Run:
    """any-llm alone, through the facade's own clients, so the transport is the same."""
    llm = facade()
    client = llm.client(model.provider)
    bodies = [build(model, h, False)[1] for h in histories]
    send = client.aresponses if model.provider == "openai" else client.amessages
    return await baseline(name, [lambda b=b: send(model=model.model, **b) for b in bodies], llm.aclose)


async def failover_native(name: str) -> Run:
    """Each request tries the missing model, then Haiku, by hand."""
    sdk, run = Native(), Run(name)
    bodies = [(body(GONE, text(n)), body(HAIKU, text(n))) for n in range(N)]

    async def one(gone: dict, haiku: dict):
        try:
            return await sdk.send(GONE, gone)
        except Exception:
            run.failed_tries += 1
            return await sdk.send(HAIKU, haiku)
    return await baseline(name, [lambda g=g, h=h: one(g, h) for g, h in bodies], sdk.aclose, run)


async def workloads() -> list[tuple[str, list[Run]]]:
    """Per provider and workload: ours and the SDK's in ABBA order, so drift hits both alike. Text adds any-llm."""
    tables = []
    for model in (NANO, HAIKU):
        llm = facade()
        try:
            followed = await follow_up(llm, model)
        finally:
            await llm.aclose()
        for label, history in (("text", text), ("text + image", image), ("follow-up", followed)):
            histories = [history(n) for n in range(N)]
            ours, theirs = (lambda: llm_master("llm_master", [model], histories),
                            lambda: native(f"{model.provider} SDK", model, histories))
            runs = [await ours(), await theirs(), await theirs(), await ours()]
            if label == "text":
                runs += [await any_llm("any-llm", model, histories), await any_llm("any-llm", model, histories)]
            tables.append((f"{model.provider} {model.model}, {label}", runs))
    return tables


async def failovers() -> list[Run]:
    histories = [text(n) for n in range(N)]
    return [await llm_master("llm_master", [GONE, HAIKU], histories),
            await failover_native("SDKs, by hand"),
            await llm_master("llm_master, per_minute=250", [GONE, HAIKU], histories, per_minute=N)]


async def first_tokens() -> tuple[list[float], list[float]]:
    """Seconds to the first text, per stream, alternating the paths so drift hits both alike."""
    llm, sdk = facade(), Native()
    ours, theirs = [], []
    try:
        for n in range(FIRST_TOKENS):
            began, first = time.monotonic(), None
            async for chunk in await llm.query(NANO, text(n), stream=True):
                if first is None and chunk["type"] == "text-delta":
                    first = time.monotonic() - began
            ours.append(first or float("nan"))
            began, first = time.monotonic(), None
            async for event in await sdk.send(NANO, body(NANO, text(n), stream=True)):
                if first is None and event.type == "response.output_text.delta":
                    first = time.monotonic() - began
            theirs.append(first or float("nan"))
    finally:
        await llm.aclose()
        await sdk.aclose()
    return ours, theirs


HEADER = ("| run | wall (s) | p50 (s) | p95 (s) | max (s) | errors | failed tries | peak connections |\n"
          "|---|---|---|---|---|---|---|---|")


def mean_p50(runs: list[Run], name: str) -> float:
    return statistics.mean(r.p50() for r in runs if r.name == name)


async def main():
    print(f"# async_query, {N} requests per run, all in flight\n")
    for title, runs in await workloads():
        print(f"## {title}\n\n{HEADER}")
        for run in runs:
            print(run.row())
        sdk = next(r.name for r in runs if r.name.endswith("SDK"))
        print(f"\nllm_master's p50 minus the SDK's: {mean_p50(runs, 'llm_master') - mean_p50(runs, sdk):+.3f} s\n")
    print(f"## A missing model, then Haiku\n\n{HEADER}")
    for run in await failovers():
        print(run.row())
    ours, theirs = await first_tokens()
    print(f"\n## Streaming time to first token, {NANO.model}, {FIRST_TOKENS} streams each\n")
    print("| path | p50 (s) | p95 (s) |\n|---|---|---|")
    for name, times in (("llm_master", ours), ("openai SDK", theirs)):
        print(f"| {name} | {statistics.median(times):.2f} | {statistics.quantiles(times, n=20)[18]:.2f} |")


if __name__ == "__main__":
    asyncio.run(main())
