# llm_master v2 — API spec

A lite wrapper around [any-llm](https://github.com/mozilla-ai/any-llm), installed as a git
dependency on our fork's main (`github.com/JamMaster1999/any-llm@main` = upstream + our three
fixes) until upstream merges.

## Build order

Each step builds on the previous ones and borrows their tests.

1. **Providers** — the per-(provider, model) dict, RPM + pricing hints. Pure schema, zero dependencies.
2. **Facade + persistent clients** — one held `AnyLLM.create(provider)` per provider. Small, everything sits on it.
3. **Call paths** — completion / messages / responses / image / speech as thin passthroughs, streaming included (usage always on, abort that works). First end-to-end calls; the live seeds become tests here and their captured responses become the golden fixtures later steps borrow.
4. **Cost** — pure function usage → dollars via genai-prices, attached as `response.cost` (and at stream end). Tests reuse step 3's fixtures, plus cost assertions.
5. **Rate limiter** — the local sliding-window bucket, then wired into calls. Self-contained unit; sits before retries because `retry_after` and the bucket must cooperate.
6. **Retries** — jittered, `retry_after`-aware loop around one candidate; SDK retries forced to 0. Tests reuse step 3's fake providers, now made to fail.
7. **Fallback chains** — candidate iteration sharing the retry budget, plus per-candidate history hygiene. The hardest piece; borrows the whole retry suite and adds cross-dialect cases.
8. **async_query fan-out** — semaphore + progress callback over the finished single-call machinery. The gradescope contract wraps the final call shape once, no rework.
9. **Telemetry** — PostHog events observing the finished pipeline (model, tokens, latency, cost, retry/fallback outcome, errors). Last because its events want everything above to exist; tested by replaying earlier suites into a capture sink.

Gemini media (nano-banana, Imagen) is out of scope until upstream answers
[mozilla-ai/any-llm#1295](https://github.com/mozilla-ai/any-llm/issues/1295); the fix lands in
any-llm, not here.

Code interpreter is deferred to its own design round. any-llm does not standardize it — three
provider-native shapes, and gemini's `executable_code` parts fall into the same converter gap as
#1295. Options on the table: provider-native per dialect (the known mess), our own sandbox behind
a normal tool call (keeps the standardized tool loop), or a separate harnessed agent (claude code
sdk / codex), which opens conversation-forking and resume questions. Nothing decided. Note: v1's
`code_execution.py` rode vertex express mode, which v2 drops — a migration dependency to resolve
in that round.

## Not rebuilt

any-llm covers these natively; the wrapper passes them through: tool calling, thinking-block
replay, structured outputs, PDF inputs, web search, typed errors (`ANY_LLM_UNIFIED_EXCEPTIONS=1`),
provider-native batch.

---

## 1. Providers

One Python file inside the package. It holds four facts per provider: rate limits, the name map
genai-prices needs, which API it speaks, and its web-search stance. It holds no prices, no model
allowlist, no API keys.

```python
from pydantic import BaseModel, Field, model_validator


class ModelLimit(BaseModel, frozen=True, extra="forbid"):
    name: str
    requests: int | None = Field(None, gt=0)   # None = inherit from provider
    interval: int | None = Field(None, gt=0)   # seconds

    @model_validator(mode="after")
    def _overrides_something(self):
        if self.requests is None and self.interval is None:
            raise ValueError(f"{self.name}: overrides nothing")
        return self


class Provider(BaseModel, frozen=True, extra="forbid"):
    name: str
    api: Literal["completion", "messages", "responses"]  # required: which API it speaks
    search: Literal["tool", "none"]  # required: its web-search stance (card 10)
    requests: int = Field(gt=0)
    interval: int = Field(gt=0)                # seconds
    models: tuple[ModelLimit, ...]             # exceptions only; () means none
    price_id: str | None = None                # genai-prices name, only when it differs

    @model_validator(mode="after")
    def _no_duplicate_models(self):
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate model entries")
        return self


_GEMINI_MODELS = [
    ModelLimit(name="gemini-2.5-flash-lite", requests=30000),
    ModelLimit(name="gemini-2.5-flash-lite-preview-09-2025", requests=30000),
    ModelLimit(name="gemini-3.1-flash-lite", requests=30000),
    ModelLimit(name="gemini-3.1-flash-lite-preview", requests=30000),
    ModelLimit(name="gemini-3.5-flash", requests=20000),
    ModelLimit(name="gemini-3-flash-preview", requests=20000),
    ModelLimit(name="gemini-2.5-pro", requests=2000),
    ModelLimit(name="gemini-3-pro-preview", requests=2000),
    ModelLimit(name="gemini-3.1-pro-preview", requests=2000),
    ModelLimit(name="gemini-2.5-flash-image", requests=5000),
    ModelLimit(name="gemini-3-pro-image", requests=2000),
]

PROVIDERS = [
    Provider(
        name="openai",
        api="responses",
        search="tool",
        requests=10000,
        interval=60,
        models=[
            ModelLimit(name="gpt-5.1", requests=15000),
            ModelLimit(name="gpt-5.2", requests=15000),
            ModelLimit(name="gpt-5.4", requests=15000),
            ModelLimit(name="gpt-5.5", requests=15000),
            ModelLimit(name="gpt-5.6-sol", requests=15000),
            ModelLimit(name="gpt-5.6-terra", requests=15000),
            ModelLimit(name="gpt-5.6-luna", requests=15000),
            ModelLimit(name="chatgpt-4o-latest", requests=5000),
            ModelLimit(name="gpt-audio-mini", requests=5000),
            ModelLimit(name="gpt-4o-audio-preview", requests=5000),
            ModelLimit(name="omni-moderation-latest", requests=1000),
            ModelLimit(name="gpt-image-1", requests=50),
        ],
    ),
    Provider(name="anthropic", api="messages", search="tool", requests=4000, interval=60, models=[]),
    Provider(name="gemini", api="completion", search="tool", price_id="google", requests=10000, interval=60, models=_GEMINI_MODELS),
    Provider(name="vertexai", api="completion", search="tool", price_id="google", requests=10000, interval=60, models=_GEMINI_MODELS),
    Provider(name="fireworks", api="completion", search="none", requests=600, interval=60, models=[]),
]

_BY_NAME = {p.name: p for p in PROVIDERS}
assert len(_BY_NAME) == len(PROVIDERS), "duplicate provider names"


def limit(provider: str, model: str) -> tuple[int, int]:
    """(requests, interval) for one bucket. A model override replaces the provider default."""
    p = _BY_NAME[provider]  # unknown provider fails loud, on purpose
    m = next((x for x in p.models if x.name == model), None)
    return (
        m.requests if m and m.requests else p.requests,
        m.interval if m and m.interval else p.interval,
    )
```

Rules:

- A model override replaces the provider default. One bucket per (provider, model), no
  provider-wide total.
- Unknown model: the provider default applies, so new models work day one with no library update.
  A typo'd name fails at the provider with a typed error, before billing.
- Unknown provider: `limit()` raises.
- Prices live in genai-prices, with its auto-refresh on (details in section 4). `price_id` is the
  only pricing fact here.

Validation:

- The editor flags wrong types, missing fields, and misspelled fields at save (pydantic + Pylance).
- The value rules (positive numbers, duplicates, rows that override nothing) run as repo tests
  before committing.

## 2. Facade + persistent clients

(to spec)

## 3. Call paths

(to spec)

## 4. Cost

(to spec)

## 5. Rate limiter

(to spec)

## 6. Retries

(to spec)

## 7. Fallback chains

(to spec)

## 8. async_query fan-out

(to spec)

## 9. Telemetry

(to spec)
