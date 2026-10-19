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
5. **Rate limiter** — a local sliding window per (provider, model), its per-minute cap set by the caller on `LLM`, wired in before retries. Done.
6. **Retries** — a jittered, `retry_after`-aware loop around one candidate, its waits capped at 2 minutes. Done.
7. **Fallback chains** — a loop over candidates. Retries exhausted, or any error that skips them, advances to the next. Done.
8. **async_query fan-out** — a semaphore over the finished single-call machinery, yielding each result as its request finishes. Done.
9. **Telemetry** — PostHog events over the finished pipeline.

## Decisions

**Models.** Recent models only: the Claude 5.5 series, GPT-5.5 and later, Gemini 3 and later.
On fireworks: GLM, DeepSeek and Kimi, at lower priority. Rules that only older models need are
not written.

**Providers (built).** providers.py is pure data: name-keyed rows, one field per line, no
code. The rosters (`API`, `ProviderName`, `SDK`) and the row's TypedDict sit at its top, and
every type in the repo derives from them. registry.py holds `price_id()`
and import-time asserts. The editor flags a bad row as you type; the asserts
catch it at import. The page lists no models, so a new model works the day it ships. Unknown
providers raise. vertexai uses any-llm's first-class provider with full GCP login; v1's express
mode is gone.

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

**Rate limits.** We keep no table of the providers' limits: they differ per account and tier,
and they change without notice. The caller caps its own tries instead. `LLM(per_minute=250)`
lets at most 250 tries start in any 60 seconds per (provider, model), across every call through
that `LLM`. With no cap, the default, nothing waits.

- A sliding window per (provider, model), shared by `query()`, streams, `image()` and `speech()`.
  Most providers count their limits per model too. One window for every model would make a fallback
  wait out its failed primary's tries: measured live on 2026-10-19, 250 calls whose primary
  failed at once waited 61 seconds for their fallback under one shared window of 250.
- Every try waits for a slot in its model's window before it sends, retries and fallback hops
  included, which is why the limiter sits before retries. A moderation check takes no slot: it
  isn't a candidate's try.
- The wait counts toward `response.latency` but not an attempt's latency, which starts with the
  try.
- The window is local to one `LLM`, so other processes don't share it. A provider's 429 still
  surfaces typed, for the retries.
- A cap below 1 raises `ValueError`, since nothing could ever start.

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
- Between tries, a random wait of up to 0.5 × 2ⁿ⁻¹ seconds before try n + 1 (full jitter), so
  7.5 seconds at most over a candidate's five tries.
- A `RateLimitError` (a 429) waits as long as the provider asks:
  - its `Retry-After`, or Gemini's `retryDelay`, which Gemini sends in the 429's body instead.
    any-llm doesn't read `retryDelay` yet, so the pipeline does until
    [mozilla-ai/any-llm#1622](https://github.com/mozilla-ai/any-llm/pull/1622) ships;
  - without either, full jitter, like any other transient failure.
  - Checked live on 2026-10-19: Gemini sends no `Retry-After` and no rate-limit headers,
    Fireworks only `x-ratelimit-over-limit`, and Anthropic's spend-cap 429 no `retry-after`.
- A candidate's waits between tries add up to 2 minutes at most. A wait that would go past that
  ends its tries instead, so a `Retry-After` of 60 seconds allows three tries, and one of 300 ends
  the candidate at once. Only the waits count, never the time inside a try or the limiter's.
- A genai request (gemini, vertexai) that sets no `timeout`, in its config or its `extra`, fails a
  try after 600 seconds with nothing new: no answer, or no next chunk of a stream. The timer wraps
  each read from the provider, so a stall mid-stream yields `reset-step` like any other failure.
  genai sets no timeout of its own, and its `timeout` would cap a whole stream, cutting a long
  agent turn. The other SDKs already end a read after 600 quiet seconds.
- Every failed try's error lands, in order, on its candidate's `Attempt.errors`. There every
  error in its chain drops its frames' locals and the HTTP request and response the SDK kept, so a
  kept error holds no copy of the request and its media. Request ids, statuses and codes stay. An
  attempt's latency sums its tries; the waits count only toward the call's latency.
- A streamed try that fails after its `start-step` yields `reset-step`, and the next try streams
  from the start under a new `start-step`.
- When a candidate's last try fails, its error goes to fallback (step 7).

Fallback (step 7), the same for every call and stream:

- A call takes one candidate or a list, tried in order. An empty list raises `ValueError` before
  anything is sent.
- When a candidate's error skips the retries, its fifth try fails, or its next wait would pass the
  2-minute budget, the next candidate starts afresh: five tries and its own 2 minutes of waits.
  Every error moves on, typed or not. A cancelled call stops.
- Each hop builds the request again for the new candidate, so history replays per its own origin
  rule (IO.md).
- `attempts` holds one entry per candidate tried, in order, each with its own errors. The answering
  candidate is last, and it sets `origin`, `usage` and `cost`.
- A stream that fails over mid-answer yields `reset-step`, then the next candidate streams from
  the start under a new `start-step`.
- When every candidate fails, the call raises the last candidate's last error, typed as a single
  candidate's would be. Notes on it list every earlier error, one per try.

**Fan-out.** `llm.async_query(candidates, requests, *, concurrency=None)` runs many `query()`
calls at once. It yields `(index, result)` as each request finishes: the request's position, and
its `Response` or the error it raised after its last candidate. One failure never stops the
others, so one unreadable image can't sink a batch.

```python
async for index, result in llm.async_query(candidates, requests, concurrency=50):
    if isinstance(result, Exception):
        ...  # this request failed; the rest go on
```

- Each request is a dict: `messages`, and `moderation`, False by default. Any other key raises
  `ValueError` before anything is sent: config and fallbacks live on the candidates, and a v1
  request's `kwargs` or `fallback_*` dropped silently would change the answers.
- At most `concurrency` requests run at once, 128 when it's None; below 1 raises `ValueError`. The
  rate limiter still gates every try.
- Each request runs the whole pipeline: moderation, retries, fallback and cost. Its errors,
  validation and moderation included, are its result.
- Results come in the order requests finish. v1's list in request order and its
  `progress_callback` are gone: the loop itself shows progress, and the index places each result.
- Leaving the loop early cancels the requests still running. Inside `contextlib.aclosing`, the
  exit also waits for them; after a bare `break`, Python closes the iterator, and so cancels and
  waits, one event-loop turn later.
- An empty list yields nothing and sends nothing.
- Each request in flight holds one pooled connection, since aiohttp speaks HTTP/1.1. The OpenAI
  and Anthropic SDKs' aiohttp clients allow 1000 connections each (Fireworks rides OpenAI's), and
  google-genai's has no limit, so the pool never caps `concurrency` below 1000.

**Money.** genai-prices converts usage to dollars, with its auto-refresh on. `price_id` in
the provider row is the only pricing fact we keep. Batch calls apply the provider's 50%
discount manually.

- `cost` on a `Response` or a `Media` is dollars, a `float`. The pipeline prices the answering
  try's usage with genai-prices' `calc_price` when the call ends, streams included.
- It is None when usage is None, when genai-prices doesn't know the model, or when a price entry
  can't split the usage: unknown, never zero, and the answered call still returns.
- Only the answering try is priced, so a failed try's spend goes uncounted. That includes replies
  the provider billed in full, a `CutCall` or a `MissingMedia`: fallback moves past them, so an
  answered call can cost more than its `cost`.
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
  - A Gemini call with audio input and a cache hit costs None. any-llm drops Gemini's split of the
    cached tokens by modality, and on a model with its own cached-audio rate, such as
    gemini-3.1-flash-lite, genai-prices won't guess how much of the cache read was audio.
  - OpenAI's flex and priority tiers price at the standard rate. genai-prices can price them, but
    no caller sets `service_tier` yet.

**Telemetry.** PostHog is the only sink; Xano error reporting is dropped. Events carry
model, tokens, latency, cost, retry and fallback outcome, and errors.

**Tests.** Live-first: goldens pin request shapes, live calls prove behavior. Recorded provider
exchanges replay offline and pin every request body; `RECORD=1` re-records them live. Two
benchmark gates run live, outside the test suite, in `bench/gates.py`: `async_query` with 250
requests in flight, plain and failing over, and streaming time to first token. Each sends the same
requests straight through any-llm on the same transport, so the gap is our code. On 2026-10-19
llm_master's p50 came within 0.1 seconds of any-llm's, at 250 connections in flight, and its first
token came no later.

## Not rebuilt

any-llm covers these natively, and the wrapper passes them through: tool schemas,
structured outputs, PDF and audio inputs, cache-token usage, typed errors
(`unified_exceptions=True` at create), provider-native batch.

## Deferred

- Provider-hosted tools are not supported. Every tool is ours (TOOLS.md) and runs beside
  llm_master, not under it. Coding gets its own design round.
- Consumer migration is a later phase. Old main stays frozen until the last consumer moves.
- A v2 README.
