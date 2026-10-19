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
2. **Facade + persistent clients** — one held `AnyLLM.create(provider, unified_exceptions=True, max_retries=0)` per provider. Everything sits on it. Done.
3. **Call paths** — `query()` over completion / messages / responses, plus `image()` and `speech()` (Media below), streaming included for `query()`. The live seeds become tests here, and their captured responses become golden fixtures. Done.
4. **Cost** — usage → dollars via genai-prices, attached as `response.cost`, streams included.
5. **Rate limiter** — a local sliding-window bucket per (provider, model), wired in before retries.
6. **Retries** — a jittered, `retry_after`-aware loop around one candidate.
7. **Fallback chains** — a loop over candidates. Retries exhausted or a typed error advances to the next.
8. **async_query fan-out** — semaphore + progress callback over the finished single-call machinery. The v1 signature survives; consumers depend on it.
9. **Telemetry** — PostHog events over the finished pipeline.

## Decisions

**Models.** Recent models only: the Claude 5.5 series, GPT-5.5 and later, Gemini 3 and later.
On fireworks: GLM, DeepSeek and Kimi, at lower priority. Rules that only older models need are
not written.

**Providers (built).** providers.py is pure data: name-keyed rows, one field per line, no
code. The rosters (`API`, `ProviderName`, `SDK`) sit at its top, and every type in
the repo derives from them. shapes.py holds the row TypedDicts. registry.py holds `limit()`
and import-time asserts. The editor flags a bad row as you type; the asserts
catch it at import. A model override replaces the provider default. Unknown models inherit
the provider default, so new models work day one. Unknown providers raise. vertexai uses
any-llm's first-class provider with full GCP login; v1's express mode is gone.

**Calls.** The caller states provider and model; there is no model-name routing. Fallbacks
are explicit candidate lists on the call. One history format, our own (IO.md), goes in, and
assembly spells it per dialect on every call. A model's own turns replay to it verbatim;
every other model sees their text and tool calls, never their reasoning. Unset config always
means the provider default.
A field with no spelling on an API raises typed `UnsupportedField`. The facade holds one
client per provider, so every call reuses its connection pool. Streams always report usage
and can abort. `query()` never runs a tool: it returns the model's tool calls, and the loop
belongs to the caller.

**Media.** `query()` is conversation only: text and tool calls, no media out. Images and
speech get their own calls, `image(candidates, prompt, images)` and `speech(candidates, text)`,
on the same pipeline as `query()` — limiter, retries, fallback, cost, telemetry. Each returns a
`Media`: the file's bytes, its mime type, the candidate that made it, usage, cost, attempts and
latency. Gemini, Vertex AI and OpenAI make both; Anthropic and Fireworks raise
`UnsupportedField`. A media candidate takes only `timeout` from its config; the provider's own
options (size, quality, format, voice) ride its `extra`, and OpenAI speech uses the `alloy` voice
unless `extra` names one. A candidate that answers without the media raises `MissingMedia`, so
fallback moves on.

An edit is lineage, not a conversation: the parent image plus the new instruction, one
request, no history and no server state. Any provider can edit any provider's image.

Live-checked 2026-10-18: 
* openai doubles the nucleus and keeps the rest; a gemini-made image edits cleanly on openai; gemini edits keep the image but understate the change. 
* The media tools own the files and the lineage record (TOOLS.md). 
* OpenAI speech reports its usage only in its streamed format, which `speech()` asks for unless `extra` sets `stream_format`.

**Failures.** SDK transport retries are off (`max_retries=0` at create, honored by all five
providers); the wrapper owns all retries. Each candidate gets its full retry budget — 3 candidates × 5 retries = 15 tries, by design. Any typed error hands off to the next candidate, and a fallback hop just re-runs assembly. A
timeout expiry is a typed `ProviderError`, retryable like the rest.

Known limit: gemini and vertexai keep google-genai's aiohttp transport for async throughput,
and that transport retries a connection error once on its own, after 1–10 seconds, ignoring
`max_retries`. HTTP errors are unaffected. A drop after the server received the request can
bill one generation twice
([googleapis/python-genai#3160](https://github.com/googleapis/python-genai/issues/3160)).

**Money.** genai-prices converts usage to dollars, with its auto-refresh on. `price_id` in
the provider row is the only pricing fact we keep. Batch calls apply the provider's 50%
discount manually.

- `cost` on a `Response` or a `Media` is dollars, a `float`. The pipeline prices the answering
  try's usage with genai-prices' `calc_price` when the call ends, streams included.
- It is None when usage is None or genai-prices doesn't know the model: unknown, never zero.
- Only the answering try is priced. A failed try reports no usage, so its spend goes uncounted.
- The genai-prices provider is the row's `price_id`, else the provider's own name. An import-time
  check rejects a name genai-prices doesn't know, so a typo can't pass for an unpriced model.
- `Usage` breaks out the parts providers price apart. Each is part of `input` or `output`:
  - `cache_write_1h`: Anthropic's 1-hour cache writes, 2x the input rate against 1.25x for
    5 minutes.
  - `input_audio`, `input_image`, `output_audio`, `output_image`: audio and image tokens. OpenAI's
    audio input costs 16x its text rate, and a Gemini image's output 20x.
- `LLM()` starts genai-prices' hourly refresh, one thread for every `LLM` in the process, and
  `aclose()` stops it. `LLM(update_prices=False)` starts none, so costs come from the prices
  genai-prices holds: its bundled snapshot, unless a refresh ran. The tests' `llm` fixture uses
  it, so a replayed call never fetches prices.
- Known limits:
  - The bundled snapshot lags. On 2026-10-19 it lacked Fireworks' glm-5p3 and deepseek-v4p1-flash
    and priced gemini-3.8-flash-tts as gemini-3.8-flash; the refreshed prices had all three.
  - OpenAI's flex and priority tiers price at the standard rate. genai-prices can price them, but
    no caller sets `service_tier` yet.

**Telemetry.** PostHog is the only sink; Xano error reporting is dropped. Events carry
model, tokens, latency, cost, retry and fallback outcome, and errors.

**Tests.** Live-first: goldens pin request shapes, live calls prove behavior. Recorded provider
exchanges replay offline and pin every request body; `RECORD=1` re-records them live. Two
benchmark gates run before migration: conversion overhead at concurrency 50, and streaming time
to first token.

## Not rebuilt

any-llm covers these natively, and the wrapper passes them through: tool schemas,
structured outputs, PDF and audio inputs, cache-token usage, typed errors
(`unified_exceptions=True` at create), provider-native batch.

## Deferred

- Provider-hosted tools are not supported. Every tool is ours (TOOLS.md) and runs beside
  llm_master, not under it. Coding gets its own design round.
- Consumer migration is a later phase. Old main stays frozen until the last consumer moves.
- A v2 README.
