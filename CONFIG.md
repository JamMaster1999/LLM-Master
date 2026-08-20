# The `llm.query` config

One request, one config. The caller writes each field once; for every fallback candidate, the
wrapper translates the fields into that candidate's dialect:

- **completion** — any-llm's `acompletion`, the openai-chat shape. Our route for gemini, vertexai,
  fireworks.
- **messages** — any-llm's `amessages`, anthropic's native shape. Anthropic only.
- **responses** — any-llm's `aresponses`, openai's newer API. Openai only.

The config is a pydantic model: `Literal` types for the enums, a docstring per field, so the editor
autocompletes every field, flags bad values as you type, and shows the docs on hover. Fallback
candidates are typed the same way. Field order below is the authoring order.

1. `system` — how the model should behave
2. `messages` — what you're sending
3. `stream` — deliver as a stream
4. `modalities` — output kinds: text / image / audio
5. `temperature` — randomness
6. `thinking` — none / low / medium / high / xhigh
7. `thinking_summaries` — surface the reasoning text or not
8. `tools` — what it may call
9. `tool_choice` — auto / none / forced / parallel on-off
10. `web_search` — provider-native search
11. `response_format` — structured output schema
12. `max_tokens` — output cap
13. `timeout` — per-request seconds
14. `cache` — auto / none / minutes
15. `moderation` — pre-send openai moderation

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

The shape is openai's: each message is a role plus content.

- Roles: `user`, `assistant`, `tool`. System-role entries are rejected (field 1's rule).
- Content: a string, or a list of `text` / `image_url` / `file` (pdf) parts. Nothing else; unknown
  part types fail validation. v1's five alternate message shapes are gone.
- Typed both ways: plain openai-shaped dicts validate through pydantic, and typed part classes
  exist for autocomplete. Same schema either way.
- Images: we accept a data-uri, a URL, or a local file path — and always convert to a base64
  data-uri before sending, downloading and converting (HEIC→PNG) as needed. One code path for all
  providers; no provider is ever handed a URL.
- PDFs: the `file` part with a base64 data-uri, live-verified on anthropic, openai, and gemini.
  Audio and video input exist nowhere in any-llm today — those parts get added when that changes.
- Tool results: openai shape (`{"role": "tool", "tool_call_id": ..., "content": ...}`), per
  field 8.

Assistant turns — the two rules that make history work:

- An assistant turn may carry its provider's native blocks: anthropic's signed thinking, openai's
  reasoning items, gemini's signature pouch. We store them untouched and echo them untouched to
  the same provider. This one rule buys both reasoning replay and cache stability.
- On fallback to a different provider, foreign native blocks are deleted, never translated.
  Translating another model's reasoning is the LiteLLM bug; deleting is the only correct move.

Translation per dialect: completion — handed over as-is, each provider's converter consumes this
shape. messages — any-llm's own openai→anthropic converter, reused. responses — our rename
(`text`→`input_text` and friends); native items pass verbatim.

The round trip: the response's history-ready form appends straight onto this list, so
save → reload → next call is lossless by construction.

## 3. `stream` — `bool`, default False

- completion: `stream=True`, and we always inject `stream_options={"include_usage": true}` —
  without it, fireworks streams end with no usage at all (live-verified; gemini sends usage
  regardless).
- messages: anthropic's native event stream; usage arrives in its closing events.
- responses: `stream=True` on the event stream.
- Return type changes: off → one response object; on → an async iterator of stream events. The
  exact event shape gets specced together with the response type.
- Known limit: a stream the caller kills mid-flight records zero usage, so that one generation
  logs no cost.

## 4. `modalities` — `list["text" | "image" | "audio"] | None`, None = text

Today, any non-text value raises our typed unsupported error on every dialect. Per dialect, why:

- completion, gemini: the request side works (`response_modalities=["TEXT", "IMAGE"]`), but
  any-llm still throws away the media parts of the response
  ([#1295](https://github.com/mozilla-ai/any-llm/issues/1295)) — you'd be billed for an image you
  never receive. Until the fix lands, refusing beats silently charging.
- completion, fireworks: no such capability exists.
- messages, anthropic: no output modalities at all. Permanent.
- responses, openai: audio lives on their chat dialect, and images (gpt-image-2) need
  verification before their arms open.

The field is the stable config; the arms unlock one by one: gemini image and audio when #1295
merges; openai audio by routing those candidates to the completion dialect; openai images once a
live seed verifies the wire shape and that any-llm preserves the returned images.

The modality comes from picking the right model, not just setting the field. Once an arm is open,
we translate and send — a model that rejects the ask fails typed (live: gemini-2.5-flash 400s
"This model only supports text output"); a model that silently ignores it returns what it returns
(live: gemini-3.5-flash ignores IMAGE and answers in text). No error means no error: no warnings,
no response inspection, no model list. A candidate that raises hands off to the next candidate
like any other typed error.

## 5. `temperature` — `float | None`, default None

- All three dialects: passed through untouched. None → omitted, provider default applies.
- Ranges differ (anthropic tops at 1.0, the others at 2.0) and reasoning models reject the
  parameter outright (live: claude-sonnet-5 400s "`temperature` is deprecated for this model").
  All of that is the provider's call: their 400 surfaces typed, fallback moves on. No clamping,
  no model knowledge.
- Different temperatures per provider are what per-candidate overrides are for.
- Our tests prove the translation, not consumer values; a chain whose candidates always fail
  shows up as error events in telemetry.

## 6. `thinking` — `"none" | "low" | "medium" | "high" | "xhigh" | None`, None = provider default

- completion: `reasoning_effort=` passed through; any-llm owns the per-provider mapping. `none`
  genuinely disables where possible; thinking-only models surface the provider's own 400 typed;
  gemini 3.5+ clamps to MINIMAL with a logged warning
  ([#1294](https://github.com/mozilla-ai/any-llm/pull/1294), carried on the fork until merged).
  `xhigh` maps to gemini's top tier.
- messages: `none` → `thinking: {"type": "disabled"}`; levels → `thinking: {"type": "adaptive"}`
  plus `output_config: {"effort": level}`. Anthropic's named tiers map 1:1, no budget table.
  Live-verified on claude-sonnet-5: adaptive levels produce thinking blocks, disabled produces
  none.
- responses: `reasoning={"effort": ...}` passed through, plus `summary: "auto"` on every level —
  summaries are always fetched (openai's thought text is opt-in), with the view governed by
  thinking_summaries. Tiers a model doesn't take come back as openai's rejection, typed.
- No capability list: providers that map a tier clamp to nearest, providers that reject fail
  typed, and consumer smoke runs catch config rot in CI. Adoption note: any-llm sometimes
  silently drops `none` (xai does), so "does none really disable" is a one-time check when
  adopting a provider — our six are verified honest.

## 7. `thinking_summaries` — `bool`, default False

- A pure view flag — no request-side translation on any API. Thought text is always fetched when
  thinking is on: gemini via any-llm's hardwired `include_thoughts=True`, anthropic's full blocks
  by its own design, openai via `summary: "auto"` riding thinking's responses arm. io filters
  what the readable view exposes, uniformly. Cost is unchanged everywhere — thought tokens are
  billed whether or not you look at them.
- Live note, openai: a summary only materializes when there's enough reasoning to summarize; an
  empty summary list is not an error.
- The flag governs only the readable view. Native blocks, signatures included, always survive in
  the conversation history — the tool loop echoes them back and anthropic verifies the signature
  server-side; a stripped block 400s the conversation mid-loop.
- Two-view rule, which lands in the response type and in `messages`: the response exposes a
  history-ready native form for server-side saving — full fidelity, replayable — and readable
  fields for forwarding to the frontend. Saved history round-trips through `messages` unchanged;
  cross-provider fallback does its history hygiene at candidate translation.
- On with thinking off: quiet no-op.

## 8. `tools` — list of python callables or openai-format dicts, default None

- One standard, both directions: write tools in openai's function format or as bare callables;
  append results in openai shape. Callables normalize through any-llm's own `callable_to_tool()`
  — schema built from the signature and docstring. Mixing dicts and callables is fine.
- completion: openai format is the native input; each provider's any-llm converter handles the
  whole loop.
- messages: tool schemas are our own one-rename translation (`parameters` → `input_schema`);
  tool-result messages ride any-llm's anthropic converter, consecutive-result merging included.
- responses: same information, flattened — two three-line reshapes at build (tool schemas, and
  results as `function_call_output` items).
- Scope boundary: `query()` never runs a tool. It returns the model's tool calls; executing and
  looping is the caller's job.
- Live seeds verify the loop end to end per provider — we read their converters, but we bill
  against what actually runs.

## 9. `tool_choice` — `ToolChoice(mode, only, parallel) | None`, default None

- One flat object, every field optional and mixable; unset = provider default. `mode`: `auto` /
  `none` / `required`. `only`: one name or a list — restrict the menu. `parallel`: False = one
  call per turn.
- Authoring validation: `mode="none"` with `only` or `parallel` set is rejected (nothing would
  run, so neither has anything to govern); `only=[]` likewise; a one-name list normalizes to the
  string, so `only=["f"]` and `only="f"` translate identically.
- completion + responses: bare `mode` passes as the openai string; `required` + single `only` is
  the named-function form; any other `only` is `allowed_tools` with mode auto|required (auto =
  may use only these); `parallel` splits out as `parallel_tool_calls`. completion nests the
  openai spellings, responses is flat. any-llm standardizes completion per provider; gemini's
  named / `none` / `allowed_tools` forms come from
  [mozilla-ai#1300](https://github.com/mozilla-ai/any-llm/pull/1300), carried on the fork until
  merged (an auto-mode menu on gemini stays a typed error — gemini only honors the name list in
  ANY mode).
- messages: `required`+single → `{type: tool, name}`; `required` otherwise → `any`; `none` →
  `none`; else `auto`; `parallel` folds in as `disable_parallel_tool_use`. Anthropic has no
  subset form, so a multi-name `only` collapses to the bare mode — with several allowed tools,
  let the model pick.
- None → omitted everywhere; providers default to auto when tools are present.
- Live note, anthropic: a required tool is guaranteed called, not called alone — it may add
  parallel calls of other tools. Exclusivity = `parallel=False` on top.

## 10. `web_search` — `Search(max_searches=...) | None`, default None

- Flat object: `None` = off, `Search()` = on, `Search(max_searches=n)` = capped.
- Every provider row in providers.py declares a required `SearchStance` — `"tool"` (has a search
  tool we ask for) or `"none"` (raises typed before any spelling). A provider cannot be added
  without this decision.
- completion, gemini/vertexai: the plain dict `{"google_search": {}}` rides the standard tools
  list. No cap parameter; gemini grounding runs at most one search round, so any cap trivially
  holds. Grounded answer text comes back; the citation metadata currently dies in any-llm's
  normalization.
- messages: anthropic's `web_search_20260318` server tool with `max_uses` (version live-verified)
  — full result blocks, citations, and a search count in usage. The richest return of the three.
- responses: openai's `web_search` tool, capped via `max_tool_calls`. Caveat: that caps all
  built-in tool calls collectively — the equivalence holds only while search is the only built-in
  tool we inject.
- Citations are not standardized by any-llm — two native styles (anthropic citation objects,
  openai `url_citation` annotations), gemini none. Our response type folds all three into one
  canonical citation shape. Upstream path: any-llm's `annotations` field exists unwritten — a
  gemini-first PR mapping `grounding_metadata` into it is staged on the fork; wrapper-side
  normalization covers everyone except gemini, whose data dies inside any-llm.
- Search fees: genai-prices already prices anthropic ($10/1k, extractor included). openai reports
  no count in usage: we count the search calls in the response trace times the published fee.

## 11. `response_format` — `Format(type, schema) | None`, default None

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

## 12. `max_tokens` — `int | None`, default None

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

## 13. `timeout` — `float` seconds `| None`, default None

- All three dialects take any-llm's `timeout` param. Enforcement live-verified on all three: a
  1ms timeout raises the typed error (test_trivials).
- None → the SDK's own default applies. One exception: anthropic with max_tokens also unset gets
  `timeout=600` (see `max_tokens`).
- An expiry surfaces as the typed timeout error, a retryable failure — the retry and fallback
  layers decide what happens next, not this field.

## 14. `cache` — `"auto" | "none" | int` minutes (validated 5–60), default "auto"

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
  caches nothing without one. Gemini: no lifetime control exists → typed unsupported.
- Costs, verified: openai 5.6+ writes 1.25x flat, auto mode included — an unrecoverable fee on
  one-shot prompts, which is why `"none"` exists. Anthropic writes 1.25x at 5m and 2x at 1h;
  reads 0.1x everywhere. genai-prices carries a single cache-write price, so anthropic's 1h-tier
  2x would undercount — noted for the cost step.
- Build seed: whether pre-5.6 openai models accept `prompt_cache_options` gracefully.
- Lifetime exotics beyond the field ride the candidate's `extra`.

## 15. `moderation` — `bool`, default False

- Not a translation but a pre-step: when on, the input text runs through openai's
  `omni-moderation-latest` before dispatch. A flagged input raises our typed refusal before any
  model is called or billed — same behavior on every dialect, every candidate.
- The moderation call is free (openai prices it at zero; 1000 rpm in our registry).
- v1 had this exact flag on `query()`; the chat platform is its consumer.
