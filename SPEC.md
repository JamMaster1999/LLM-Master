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
4. **Cost** — usage → dollars via genai-prices, attached as `response.cost`, streams included. Done.
5. **Rate limiter** — a local sliding-window bucket per (provider, model), wired in before retries. Done.
6. **Retries** — a jittered, `retry_after`-aware loop around one candidate. Done.
7. **Fallback chains** — a loop over candidates. Retries exhausted, or any error that skips them, advances to the next. Done.
8. **async_query fan-out** — semaphore + progress callback over the finished single-call machinery. The v1 signature survives; consumers depend on it. Done.
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
`UnsupportedField`. A media candidate takes only `timeout` from its config. One that sets another
field fails with `UnsupportedField`, and fallback moves on. The provider's own options (size,
quality, format, voice) ride its `extra`, and OpenAI speech uses the `alloy` voice unless `extra`
names one. A candidate that answers without the media raises `MissingMedia`, so
fallback moves on.

An edit is lineage, not a conversation: the parent image plus the new instruction, one
request, no history and no server state. Any provider can edit any provider's image.

Live-checked 2026-10-18: 
* openai doubles the nucleus and keeps the rest; a gemini-made image edits cleanly on openai; gemini edits keep the image but understate the change. 
* The media tools own the files and the lineage record (TOOLS.md). 
* OpenAI speech reports its usage only in its streamed format, which `speech()` asks for unless `extra` sets `stream_format`.

**Rate limits.** Each `LLM` holds one limiter: a sliding window per (provider, model), from
`registry.limit()`. At most `requests` tries may start in any `interval` seconds.

- Every try waits for a slot before it sends, retries and fallback hops included, which is why
  the limiter sits before retries. Moderation waits in its own window,
  `omni-moderation-latest`'s.
- The wait counts toward `response.latency` but not an attempt's latency, which starts with the
  try.
- The windows are local to one `LLM`, so other processes don't share them. A provider's 429 still
  surfaces typed, for the retries.
- A model without a row of its own still gets its own window, at the provider's default limit.

**Failures.** SDK transport retries are off (`max_retries=0` at create, honored by all five
providers); the wrapper owns all retries and fallback, below. A timeout expiry is a typed
`ProviderError`, retryable like the rest.

Retries (step 6), the same for every call and stream:

- A candidate gets 5 tries, so 3 candidates make at most 15, by design. A try that fails
  transiently is tried again:
  - a `RateLimitError`, `GatewayTimeoutError` or `UpstreamProviderError`;
  - a `ProviderError` with no HTTP status, a 5xx, or a 2xx: timeouts, dropped connections, cut
    streams, Gemini's answer with no choice, and Anthropic's `overloaded_error` inside a 200
    stream;
  - an OpenAI `failed` status whose code is `server_error`, `rate_limit_exceeded` or
    `vector_store_timeout`, which raises a `ProviderError`.
- Any other error skips the retries: a bad key or request (any 4xx), a content filter, an unknown
  model, a context too long, our `UnsupportedField` and `MissingMedia`, and a `CutCall`, which the
  same request would cut again. Any other OpenAI `failed` code (`invalid_prompt`, `bio_policy`, an
  image error) raises an `InvalidRequestError`, so the prompt isn't sent again.
- Between tries, the provider's `Retry-After` when a `RateLimitError` carries one, capped at
  60 seconds. Otherwise a random wait of up to 0.5 × 2ⁿ⁻¹ seconds before try n + 1 (full jitter),
  so 7.5 seconds at most over a candidate's five tries.
- Every failed try's error lands, in order, on its candidate's `Attempt.errors`. Its frames drop
  their locals there, so a kept error doesn't hold the request and its media. An attempt's latency
  sums its tries; the waits count only toward the call's latency.
- A streamed try that fails after its `start-step` yields `reset-step`, and the next try streams
  from the start under a new `start-step`.
- When a candidate's last try fails, its error goes to fallback (step 7).

Fallback (step 7), the same for every call and stream:

- A call takes one candidate or a list, tried in order. An empty list raises `ValueError` before
  anything is sent.
- When a candidate's error skips the retries or its fifth try fails, the next candidate starts
  with a fresh retry budget. Every error moves on, typed or not. A cancelled call stops.
- Each hop builds the request again for the new candidate, so history replays per its own origin
  rule (IO.md).
- `attempts` holds one entry per candidate tried, in order, each with its own errors. The answering
  candidate is last, and it sets `origin`, `usage` and `cost`.
- A stream that fails over mid-answer yields `reset-step`, then the next candidate streams from
  the start under a new `start-step`.
- When every candidate fails, the call raises the last candidate's last error, typed as a single
  candidate's would be. A note on it lists every earlier error, one line per try.

Known limit: gemini and vertexai keep google-genai's aiohttp transport for async throughput,
and that transport retries a connection error once on its own, after 1–10 seconds, ignoring
`max_retries`. HTTP errors are unaffected. A drop after the server received the request can
bill one generation twice
([googleapis/python-genai#3160](https://github.com/googleapis/python-genai/issues/3160)).

**Fan-out.** `llm.async_query(candidates, requests, *, concurrency=None, progress_callback=None)`
runs many `query()` calls at once and returns their `Response`s in request order. It keeps v1's
signature, with v2's candidates in place of v1's model name.

- Each request is a dict: `messages`, and `moderation`, False by default. Any other key raises
  `ValueError` before anything is sent: a v1 request's `kwargs` and `fallback_*` now live on the
  candidates, and dropping them silently would change the answers.
- Every request's messages validate before the first request is sent.
- At most `concurrency` requests run at once, 128 when it's None or 0, as in v1. The rate limiter
  still gates every try.
- Each request runs the whole pipeline: moderation, retries, fallback and cost.
- `progress_callback(done, total)` runs after each request finishes.
- When a request fails, the others are cancelled and its error raises, as a single call's would.
  The responses already made are lost with them.
- An empty list returns `[]` and sends nothing.

**Money.** genai-prices converts usage to dollars, with its auto-refresh on. `price_id` in
the provider row is the only pricing fact we keep. Batch calls apply the provider's 50%
discount manually.

- `cost` on a `Response` or a `Media` is dollars, a `float`. The pipeline prices the answering
  try's usage with genai-prices' `calc_price` when the call ends, streams included.
- It is None when usage is None, when genai-prices doesn't know the model, or when a price entry
  can't split the usage: unknown, never zero, and the answered call still returns.
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
  - Gemini's audio input prices at the text rate, half its price on gemini-3.1-flash-lite: the
    locked any-llm build drops Gemini's input breakdown. Fork PR #19's second commit (80f4287)
    reports it, and `input_audio` fills once that commit reaches the build.

**Telemetry.** PostHog is the only sink; Xano error reporting is dropped. Events carry
model, tokens, latency, cost, retry and fallback outcome, and errors.

- Every `query()`, `image()` and `speech()` call sends one event when it ends, answered or failed
  after its last candidate. A stream the caller leaves early sends none, like its cost (CONFIG.md
  field 3), and moderation sends none.
- The event is PostHog's LLM analytics `$ai_generation`, so its dashboards read it as is:
  - `$ai_trace_id`, new per call, and `$ai_span_name`: `query`, `image` or `speech`.
  - `$ai_provider` and `$ai_model`: the answering candidate, or the last one tried.
  - `$ai_input_tokens`, `$ai_output_tokens`, `$ai_cache_read_input_tokens`,
    `$ai_cache_creation_input_tokens` and `$ai_cache_creation_1h_input_tokens`.
    `$ai_cache_reporting_exclusive` is false, because `input` counts the cache tokens.
  - `$ai_latency`: the call's latency. `$ai_total_cost_usd`: its cost, with `$ai_cost_passthrough`
    so PostHog keeps genai-prices' price. Both cost keys are absent when the cost is unknown.
  - `$ai_stop_reason`: a query's `finish`.
  - `retries` counts the failed tries, `fallbacks` the candidates tried after the first, and
    `errors` holds one `provider/model: Type: message` line per failed try.
  - A failed call adds `$ai_is_error`, `$ai_error` with the raised error's text, and
    `$ai_http_status` when the error has one.
- No content leaves: no messages, prompts, answers or media, so a user's work never reaches
  PostHog. Events are personless (`$process_person_profile: false`).
- Telemetry is on when `POSTHOG_API_KEY` is set. `POSTHOG_HOST` overrides the host, PostHog's US
  cloud by default, where the project lives.
- Each `LLM` holds one PostHog `AsyncClient`. Events queue and go out in batches from tasks on
  the event loop, and `aclose()` sends what's left. A PostHog failure never fails a call: the SDK
  logs it.

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
