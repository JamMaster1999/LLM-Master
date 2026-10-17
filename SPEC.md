# llm_master v2 — spec

llm_master v2 is a lite wrapper around [any-llm](https://github.com/mozilla-ai/any-llm),
installed from PyPI. Old `main` stays frozen for existing consumers; this branch (`v2`)
replaces everything.

Three documents split the law. GRAMMAR.md says how config code must be shaped. CONFIG.md
rules on every config field. This file holds the build order and the project-wide decisions.

## Build order

Each step builds on the previous ones and borrows their tests. Steps 3 onward start as a
spec doc — signatures, response types, stream events — implemented only after review.

1. **Providers** — the registry data page. Done.
2. **Facade + persistent clients** — one held `AnyLLM.create(provider)` per provider. Everything sits on it.
3. **Call paths** — completion / messages / responses / image / speech as thin passthroughs, streaming included. The live seeds become tests here, and their captured responses become golden fixtures.
4. **Cost** — usage → dollars via genai-prices, attached as `response.cost`, streams included.
5. **Rate limiter** — a local sliding-window bucket per (provider, model), wired in before retries.
6. **Retries** — a jittered, `retry_after`-aware loop around one candidate.
7. **Fallback chains** — a loop over candidates. Retries exhausted or a typed error advances to the next.
8. **async_query fan-out** — semaphore + progress callback over the finished single-call machinery. The v1 signature survives; consumers depend on it.
9. **Telemetry** — PostHog events over the finished pipeline.

## Decisions

**Providers (built).** providers.py is pure data: name-keyed rows, one field per line, no
code. The rosters (`API`, `ProviderName`, `SearchStance`) sit at its top, and every type in
the repo derives from them. shapes.py holds the row TypedDicts. registry.py holds `limit()`,
`search()`, and import-time asserts. The editor flags a bad row as you type; the asserts
catch it at import. A model override replaces the provider default. Unknown models inherit
the provider default, so new models work day one. Unknown providers raise. vertexai uses
any-llm's first-class provider with full GCP login; v1's express mode is gone.

**Calls.** The caller states provider and model; there is no model-name routing. Fallbacks
are explicit candidate lists on the call. One canonical history format (openai shape) goes
in, and assembly re-shapes it per dialect on every call. Reasoning blocks never translate
across dialects; foreign ones are stripped. Unset config always means the provider default.
A field with no spelling on an API raises typed `UnsupportedField`. The facade holds one
client per provider, so every call reuses its connection pool. Streams always report usage
and can abort. `query()` never runs a tool: it returns the model's tool calls, and the loop
belongs to the caller.

**Failures.** SDK transport retries are forced to 0; the wrapper owns all retries. Each
candidate gets its full retry budget — 3 candidates × 5 retries = 15 tries, by design. Any
typed error hands off to the next candidate, and a fallback hop just re-runs assembly. A
timeout expiry is a typed `ProviderError`, retryable like the rest.

**Money.** genai-prices converts usage to dollars, with its auto-refresh on. `price_id` in
the provider row is the only pricing fact we keep. Batch calls apply the provider's 50%
discount manually.

**Telemetry.** PostHog is the only sink; Xano error reporting is dropped. Events carry
model, tokens, latency, cost, retry and fallback outcome, and errors.

**Tests.** Live-first: goldens pin request shapes, live calls prove behavior. The scripts in
live-seeds/ grow into the suite. Two benchmark gates run before migration: conversion
overhead at concurrency 50, and streaming time to first token.

## Not rebuilt

any-llm covers these natively, and the wrapper passes them through: tool calling,
thinking-block replay, structured outputs, PDF and audio inputs, image and audio outputs,
cache-token usage, web search, typed errors (`ANY_LLM_UNIFIED_EXCEPTIONS=1`), provider-native
batch.

## Deferred

- Code interpreter and agentic escalation get their own design round. A harness (Claude
  Agent SDK / Codex) would sit beside llm_master, not under it.
- Consumer migration is a later phase. Old main stays frozen until the last consumer moves.
- A v2 README.
