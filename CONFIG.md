# The `llm.query` config

One request, one config. The caller writes each field once; for every fallback candidate, the
wrapper translates the fields into that candidate's dialect:

- **completion** — any-llm's `acompletion`, the openai-chat shape. Our route for gemini, vertexai,
  fireworks. Gemini also serves responses, but only a single string input with no tools or
  history, so completion stays its route.
- **messages** — any-llm's `amessages`, anthropic's native shape. Anthropic only.
- **responses** — any-llm's `aresponses`, openai's newer API. Openai only.

The config is a pydantic model: `Literal` types for the enums, a docstring per field, so the editor
autocompletes every field, flags bad values as you type, and shows the docs on hover. Fallback
candidates are typed the same way. Field order below is the authoring order.

1. `system` — how the model should behave
2. `messages` — what you're sending
3. `stream` — deliver as a stream
4. `temperature` — randomness
5. `thinking` — none / low / medium / high / xhigh
6. `tools` — what it may call
7. `tool_choice` — auto / none / forced / parallel on-off
8. `response_format` — structured output schema
9. `max_tokens` — output cap
10. `timeout` — per-request seconds
11. `cache` — auto / none / minutes
12. `moderation` — pre-send openai moderation

Each fallback candidate additionally carries `extra: dict` — a raw pass-through door merged last
into that candidate's dialect params. Unvalidated, no autocomplete, per candidate so it can never
leak across providers. For the rare knob the config doesn't cover.

Typing rule, repo-wide: every per-provider translation is an explicit, exhaustive table — `Literal`
provider names with an `assert_never` fallthrough — so adding a provider or dialect without a
decision fails the type check instead of silently passing something through.

---

## 1. `system` — `str | None`, default None

- completion: prepended to the conversation as a system-role message.
- messages: anthropic's `system=` param.
- responses: openai's `instructions=` param.
- None → omitted everywhere.
- Companion rule: a system-role entry inside `messages` (the field) is rejected at validation —
  the `system` field is the only way to set one, so two system prompts can never fight.

## 2. `messages` — the conversation, one shape only

The shape is llm_master's own; IO.md is its law and holds every per-dialect spelling.

- Roles: `user`, `assistant`, `tool`. System-role entries are rejected (field 1's rule).
- Content: a string, or a list of text / image / file (pdf) / audio parts. Nothing else; unknown
  part types fail validation. v1's five alternate message shapes are gone.
- Typed both ways: plain dicts validate through pydantic, and typed part classes exist for
  autocomplete. Same schema either way.
- Images, PDFs, audio: a part accepts a data-uri, a URL, or a local file path and encodes it to
  base64 when it is built, downloading and converting (HEIC→PNG) as needed. History never holds a
  URL, so every call sends the same bytes and no provider is ever handed a URL.
- PDFs: live-verified on anthropic, openai, and gemini.
- Audio: base64 data plus a format name. gemini takes every format it lists (wav, mp3, aac, ogg,
  flac, m4a, opus, webm, ...); openai's audio models take wav and mp3 on the completion dialect
  only, so an openai candidate carrying audio input routes to completion. anthropic and the responses dialect have no audio input: typed unsupported.
  Live-verified on gemini and gpt-audio — a spoken sentence transcribes word for word on both.
  Video input has no part; no consumer needs it.
- Tool results: the `tool` role, carrying text, image and file parts (field 6). Fireworks takes
  text only.
- Assistant turns: a model's own turns replay to it verbatim; every other model sees their text
  and calls, never their reasoning (IO.md, the echo law).

The round trip: the response's history-ready form appends straight onto this list, so
save → reload → next call is lossless by construction.

## 3. `stream` — `bool`, default False

- completion: `stream=True`, and we always inject `stream_options={"include_usage": true}` —
  without it, fireworks streams end with no usage at all (live-verified; gemini sends usage
  regardless).
- messages: anthropic's native event stream; usage arrives in its closing events.
- responses: `stream=True` on the event stream.
- Return type changes: off → one `Response`; on → a stream of AI SDK protocol chunks whose
  `response` holds the same `Response` once it ends (IO.md, Streams).
- Known limit: a stream the caller kills mid-flight records zero usage, so that one generation
  logs no cost.

## 4. `temperature` — `float | None`, default None

- All three dialects: passed through untouched. None → omitted, provider default applies.
- Ranges differ (anthropic tops at 1.0, the others at 2.0) and reasoning models reject the
  parameter outright (live: claude-sonnet-5 400s "`temperature` is deprecated for this model").
  All of that is the provider's call: their 400 surfaces typed, fallback moves on. No clamping,
  no model knowledge.
- Different temperatures per provider are what per-candidate overrides are for.
- Our tests prove the translation, not consumer values; a chain whose candidates always fail
  shows up as error events in telemetry.

## 5. `thinking` — `"none" | "low" | "medium" | "high" | "xhigh" | None`, None = provider default

- completion: `reasoning_effort=` passed through; any-llm owns the per-provider mapping. `none`
  genuinely disables where the model has an off setting (gemini sends `thinking_budget=0`);
  models that only run with thinking on (gemini 3.x, gemini-2.5-pro) reject it before the wire
  with any-llm's typed `UnsupportedParameterError`. `xhigh` maps to gemini's top tier.
- messages: `none` → `thinking: {"type": "disabled"}`; levels → `thinking: {"type": "adaptive"}`
  plus `output_config: {"effort": level}`. Anthropic's named tiers map 1:1, no budget table.
  Live-verified on claude-sonnet-5: adaptive levels produce thinking blocks, disabled produces
  none. Most of the 5.5 generation can't turn thinking off: Opus 5.5 and Fable 5.1 reject
  `disabled`, Sonnet 5.5 too (its floor is `between_tools`, via `extra`), Haiku 5.5 takes it at
  effort `high` or below. Their 400 surfaces typed and fallback moves on — the provider's call,
  as with temperature.
- responses: `reasoning={"effort": ...}` passed through, plus `summary: "auto"` on every level —
  openai's thought text is opt-in, so summaries are always fetched. Tiers a model doesn't take
  come back as openai's rejection, typed.
- Thought text is always fetched when thinking is on: gemini via any-llm's hardwired
  `include_thoughts=True`, anthropic's full blocks by its own design, openai via `summary:
  "auto"`. It lands on `message.reasoning`; whether a user sees it is the caller's choice at
  render time. Cost is the same either way — thought tokens are billed whether or not anyone
  reads them. Live note, openai: a summary only materializes when there's enough reasoning to
  summarize; an empty summary list is not an error.
- No capability list: providers that map a tier clamp to nearest, providers that reject fail
  typed, and consumer smoke runs catch config rot in CI. Adoption note: any-llm sometimes
  silently drops `none` (xai does), so "does none really disable" is a one-time check when
  adopting a provider — our six are verified honest.

## 6. `tools` — list of python callables or openai-format dicts, default None

- Write tools in openai's function format or as bare callables; append results as `tool`
  turns (IO.md). Callables normalize through any-llm's own `callable_to_tool()` — schema built
  from the signature and docstring. Mixing dicts and callables is fine.
- Function tools only. Every tool is ours and the caller runs it (TOOLS.md). Provider-hosted
  tools — search, code execution, image generation, openai's custom tools, and the rest — fail
  validation, typed. Freeform input is a function tool with one string parameter.
- completion: openai format is the native input; each provider's any-llm converter handles the
  whole loop.
- messages: tool schemas are our own one-rename translation (`parameters` → `input_schema`);
  tool results become `tool_result` blocks in a user turn, consecutive results merged into one
  message (IO.md).
- responses: same information, flattened — two three-line reshapes at build (tool schemas, and
  results as `function_call_output` items).
- Scope boundary: `query()` never runs a tool. It returns the model's tool calls; executing and
  looping is the caller's job.
- Live seeds verify the loop end to end per provider — we read their converters, but we bill
  against what actually runs.

## 7. `tool_choice` — `ToolChoice(mode, only, parallel) | None`, default None

- One flat object, every field optional and mixable; unset = provider default. `mode`: `auto` /
  `none` / `required`. `only`: one name or a list — restrict the menu. `parallel`: False = one
  call per turn.
- Authoring validation: `mode="none"` with `only` or `parallel` set is rejected (nothing would
  run, so neither has anything to govern); `only=[]` likewise; a one-name list normalizes to the
  string, so `only=["f"]` and `only="f"` translate identically.
- completion + responses: bare `mode` passes as the openai string; `required` + single `only` is
  the named-function form; any other `only` is `allowed_tools` with mode auto|required (auto =
  may use only these); `parallel` splits out as `parallel_tool_calls`. completion nests the
  openai spellings, responses is flat. any-llm standardizes completion per provider, gemini's
  named / `none` / `allowed_tools` forms included; an auto-mode menu on gemini stays a typed
  error — gemini only honors the name list in ANY mode.
- messages: `required`+single → `{type: tool, name}`; `required` otherwise → `any`; `none` →
  `none`; else `auto`; `parallel` folds in as `disable_parallel_tool_use`. Anthropic has no
  subset form, so a multi-name `only` collapses to the bare mode — with several allowed tools,
  let the model pick.
- None → omitted everywhere; providers default to auto when tools are present.
- Live note, anthropic: a required tool is guaranteed called, not called alone — it may add
  parallel calls of other tools. Exclusivity = `parallel=False` on top.

## 8. `response_format` — `Format(type, schema) | None`, default None

- Flat object: `type` says the mode (`"text"` / `"json"` / `"schema"`), `schema` carries the
  shape (pydantic class or json-schema dict). A validator keeps them honest — `type="schema"`
  needs a schema, the other modes forbid one. `None` = unset; `type="text"` emits nothing (every
  provider's default). Pydantic reserves the name `schema`, so the field is `schema_` with
  `alias="schema"` — callers write `Format(schema=X)` either way.
- `type="json"` maps to `{"response_format": {"type": "json_object"}}` on completion and
  responses; messages raises typed (anthropic has no schema-less JSON mode). Live note: json mode
  guarantees syntax only — gemini returns numbers as strings about a third of runs, so live tests
  assert parseability and key presence, never value types.
- The schema arm stays one pydantic model, zero per-provider schemas: completion and responses
  take `response_format`, messages takes the identical `output_format` — our entire translation
  is that rename.
- completion: each provider's converter does the rest (gemini rides `response_json_schema`).
  messages: anthropic's parse path via the beta client. responses: openai's native parse.
- The parsed object surfaces on the response alongside the raw text — exact placement lands in
  the response-type round.
- Honest unknown: `stream=True` with `response_format` behaves differently per provider, and
  any-llm's parse paths may not stream at all. We pass through, rejections surface typed, and
  live seeds pin the working combinations at build.

## 9. `max_tokens` — `int | None`, default None

- Caller set it → passes through untouched: `max_tokens` on completion and messages,
  `max_output_tokens` on responses.
- Caller left it unset → completion and responses send nothing and the model's own maximum
  applies. anthropic requires the field, so its candidates get `64000` — the sonnet-line output
  ceiling, big enough that the default never clips. Models with smaller ceilings reject it with a
  typed 400; fallback moves on.
- 64k trips any-llm's long-request guard on plain non-streaming calls, so when `timeout` is also
  unset the pair ships together: `max_tokens=64000, timeout=600` — anthropic's own 10-minute
  request ceiling. Caller-set values always win. Live-verified both ways.
- Net meaning of None: "model's max" everywhere — by absence where the API allows it, by
  anthropic's own maxima where it demands a number.
- Truncation live-verified on all three: `finish_reason="length"` (completion),
  `stop_reason="max_tokens"` (messages), `status="incomplete"` + reason max_output_tokens
  (responses).

## 10. `timeout` — `float` seconds `| None`, default None

- All three dialects take any-llm's `timeout` param. Enforcement live-verified on all three: a
  1ms timeout raises the typed error (test_trivials).
- None → the SDK's own default applies. One exception: anthropic with max_tokens also unset gets
  `timeout=600` (see `max_tokens`).
- An expiry surfaces as the typed timeout error, a retryable failure — the retry and fallback
  layers decide what happens next, not this field.

## 11. `cache` — `"auto" | "none" | int` minutes (validated 5–60), default "auto"

- `"auto"`: provider behavior, no breakpoints from us — anthropic doesn't cache, openai and
  gemini auto-cache. On openai 5.6+, auto-caching bills writes at 1.25x by default; that's their
  default, not our opinion.
- `"none"`: openai sends `prompt_cache_options={"mode": "explicit"}` with no breakpoints — skips
  the cache, no write fee. Anthropic sends nothing (already off). Gemini quiet no-op: its
  auto-cache has no off switch and no write fee.
- minutes: a real lifetime, passed through — if it fails it fails. Anthropic: `5` → `"5m"`,
  `60` → `"1h"`, anything between goes literally and their API rejects unknown tiers. Openai:
  explicit mode plus ttl — only `"30m"` exists today (live-verified: other values 400 naming the
  supported set) — with one breakpoint placed at the end of the prompt, since explicit mode
  caches nothing without one. Gemini: implicit caching only, so minutes raise typed. any-llm
  can turn a `cache_control` breakpoint into an explicit gemini context cache; this field never
  emits one.
- Costs, verified: openai 5.6+ writes 1.25x flat, auto mode included — an unrecoverable fee on
  one-shot prompts, which is why `"none"` exists. Anthropic writes 1.25x at 5m and 2x at 1h;
  reads 0.1x everywhere. any-llm's usage reports reads as `cached_tokens` and writes as
  `cache_write_tokens`, with anthropic's 5m/1h split in `cache_creation_token_details`, so the
  cost step has every count it needs. genai-prices carries a single cache-write price, so
  anthropic's 1h-tier 2x needs our own multiplier — noted for the cost step.
- Build seed: whether pre-5.6 openai models accept `prompt_cache_options` gracefully.
- Lifetime exotics beyond the field ride the candidate's `extra`.

## 12. `moderation` — `bool`, default False

- Not a translation but a pre-step: when on, the input text runs through openai's
  `omni-moderation-latest` before dispatch. A flagged input raises our typed refusal before any
  model is called or billed — same behavior on every dialect, every candidate.
- The moderation call is free (openai prices it at zero; 1000 rpm in our registry).
- v1 had this exact flag on `query()`; the chat platform is its consumer.
