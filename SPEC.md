# llm_master v2 — API spec

A lite wrapper around [any-llm](https://github.com/mozilla-ai/any-llm), installed as a git
dependency on our fork's main (`github.com/JamMaster1999/any-llm@main` = upstream + our carried
fixes) until upstream merges them all and releases — then the dependency becomes a plain PyPI pin.

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

Three modules with one job each. `providers.py` is pure data — the page you edit when
providers change. It holds four facts per provider: rate limits, the name map genai-prices
needs, which API it speaks, and its web-search stance. It holds no prices, no model
allowlist, no API keys. The rosters everything derives from (`API`, `ProviderName`,
`SearchStance`) live at its top; the rows are a dict keyed by provider name.

```python
API = Literal["completion", "messages", "responses"]
ProviderName = Literal["openai", "anthropic", "gemini", "vertexai", "fireworks"]
SearchStance = Literal["tool", "none"]

PROVIDERS: dict[ProviderName, ProviderRow] = {
    "openai": {
        "api": "responses",
        "search": "tool",
        "requests": 10000,
        "interval": 60,
        "models": {
            "gpt-5.1": {"requests": 15000},
            # ... one model per line ...
        },
    },
    # ... one row per provider ...
}
```

`shapes.py` holds the `ProviderRow` / `ModelRow` TypedDicts the page is checked against.
`registry.py` holds the reads and the import-time asserts:

```python
def limit(provider: ProviderName, model: str) -> tuple[int, int]:
    """(requests, interval) for one bucket. A model override replaces the provider default."""
    row = PROVIDERS[provider]  # unknown provider fails loud, on purpose
    override = row["models"].get(model, {})
    return override.get("requests", row["requests"]), override.get("interval", row["interval"])


def search(provider: ProviderName) -> SearchStance:
    """The provider's web-search stance, from its row."""
    return PROVIDERS[provider]["search"]
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

- The editor flags an unregistered provider key, wrong types, missing fields, and misspelled
  fields as you type (TypedDict + Literal, via Pylance).
- The value rules (keys match `ProviderName`, positive numbers, overrides that override
  something) are import-time asserts in registry.py — any test run or real import trips them.

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
