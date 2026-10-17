# The message structure

One conversation, one shape. The caller writes openai's chat shape once; assembly re-spells it
into each candidate's dialect on every call, fallback hops included. This page is the law for
what a message may hold and how each dialect spells it. Config fields live in CONFIG.md; code
shape in GRAMMAR.md. The response type and stream events get their own round; this page pins
only what they must append back into the history.

## The three roles

| role | fields | meaning |
|---|---|---|
| `user` | `content` | what you send |
| `assistant` | `content`, `tool_calls`, `reasoning`, `origin`, `native` | what a model said |
| `tool` | `tool_call_id`, `content` | one tool call's result |

- `system` is not a role. The `system` config field is the only system prompt; a system-role
  entry fails validation (CONFIG.md field 1).
- `content`: user takes a string or a list of parts; assistant holds `str | None`; tool holds a
  string — the portable subset every dialect can spell. Anthropic's image tool-results and
  `is_error` flag wait until a consumer needs them.
- One `Message` class: `role` is the enum, the other fields are optional, one validator holds
  every cross-field rule. Plain openai-shaped dicts validate through it; the class exists for
  autocomplete. Same schema either way. The schema is closed — a typo fails loud, and the
  wrapper-owned keys (`reasoning`, `origin`, `native`) stay ours.

## Parts

Wire spellings are openai chat's. The set is closed — a discriminated union on `type`, unknown
types fail validation. Keys inside a part ride through each rename and the provider judges them.

- text — `{"type": "text", "text": str}`
- image — `{"type": "image_url", "image_url": {"url": <base64 data-uri>}}`
- file (pdf) — `{"type": "file", "file": {"file_data": <base64 data-uri>, "filename": str}}`
- audio — `{"type": "input_audio", "input_audio": {"data": <base64>, "format": str}}`

Authoring: `Image` and `Audio` accept a data-uri, a URL, or a local file path. One pre-pass at
call time converts everything to base64 — downloading and converting (HEIC→PNG) as needed, the
audio format read off the MIME type — before candidate iteration, so a fallback hop never
re-downloads and no provider is ever handed a URL. Validation itself does no I/O; a failed download or conversion raises typed before any
model is billed. Size stays the provider's judgment; for the record: anthropic caps images at
5MB post-encoding, gemini inlines at most 20MB per request.

## Assistant turns: readable plus native

Two views, one message.

The readable fields — `content`, `reasoning`, `tool_calls` — are what every consumer reads, what
a frontend renders after save → reload, and what a foreign candidate translates. `reasoning` is
the provider-exposed thought text (anthropic's thinking, openai's summaries, gemini's thoughts —
not equivalent disclosures, one display surface). It is a view: never a wire field, never a
translation source.

Provider-minted turns carry two more fields:

- `origin` — `{"provider": ..., "model": ...}`: the candidate that minted the turn. Stamped on
  every response, from the candidate, not from the response's model field (gemini's says
  `google/genai`).
- `native` — the provider's verbatim output, opaque:
  - anthropic: the response's `content` list — thinking (signed), redacted_thinking, text,
    tool_use, server_tool_use, web_search_tool_result, and whatever the API grows next. Stored
    as wire JSON, never re-validated against a block roster.
  - openai: the response's `output` list — reasoning (with encrypted_content), message,
    function_call, web_search_call, ...
  - gemini, vertexai, fireworks: no `native`. The chat shape is the completion dialect's
    history-ready form; gemini's thought signatures ride any-llm's inline `extra_content`
    pouches on the message and on each tool call, untouched.

**The echo law.** Assembling for a candidate: a turn whose `origin` equals the candidate —
provider and model both — replays `native` verbatim and its readable fields are ignored;
gemini's inline pouches ride under the same condition. Any other candidate gets `native` and
pouches deleted and the readable fields translated. Signed and encrypted reasoning is
model-bound: a gpt → gpt-mini or opus → sonnet hop must degrade, or it 400s. Deleting is the
only correct move — translating another model's reasoning is the LiteLLM bug. Deletion happens
per assembly; stored history is never mutated, so a later hop back to the origin replays in
full.

## Tool calls and results

One spelling both directions, openai's:

- call: `{"id": str, "type": "function", "function": {"name": str, "arguments": <json str>}}`,
  plus gemini's inline `extra_content` pouch.
- result: `{"role": "tool", "tool_call_id": str, "content": str}`.

Ids ride across providers untouched — anthropic's `tool_use.id` and openai's `call_id` land in
`id`, and results match back up on whatever candidate runs next. One exception at ingest:
any-llm mints gemini ids as `call_{hash(name)}_{index}`, which collide across turns, so gemini
turns get fresh unique ids before entering history. Gemini itself matches results by name and
never reads ids back. `query()` never runs a tool; the loop is the caller's (CONFIG.md field 8).

## Validation

One validator, every cross-field rule:

- Roles: user / assistant / tool. System-role entries rejected. The list must not be empty.
- `tool_call_id` required on tool turns, forbidden elsewhere. `tool_calls`, `reasoning`,
  `origin`, `native` are assistant-only. `native` requires `origin`.
- `reasoning` accepts any-llm's `{"content": str}` object form and stores the plain string.
- Every tool call's `arguments` must parse as JSON — anthropic and gemini need the object, and a
  broken history should fail here, not on the wire.
- The tool loop must be whole: tool turns immediately follow their assistant turn, every
  `tool_call_id` matches one of that turn's calls, no duplicates, every call answered before the
  next turn. Anthropic, openai, and gemini each 400 on a different violation; we reject once,
  before any wire.
- An assistant turn carrying nothing — no content, no tool_calls, no native — is rejected.
- A trailing text-only assistant turn is allowed and rides as-is: anthropic treats it as
  prefill, the others as context. Provider judges.

## Translation per dialect

io/messages.py follows the grammar: one case per API, every answer out of a case.
`translate(messages, api, provider, model, system)` returns the dialect's message fragment.
Provider and model drive the echo law; system rides in because completion has no parameter for
it.

### completion → `{"messages": [...]}`

Handed over as-is — each provider's any-llm converter consumes this shape.

- `system` set → one system-role message prepended.
- user, tool → verbatim, audio parts included: gemini's converter and openai's audio models both
  take the `input_audio` spelling. An openai candidate carrying audio runs this dialect
  (CONFIG.md field 4).
- assistant → `reasoning`, `origin`, `native` dropped (views and carries, never wire — fireworks
  sends unknown keys straight to theirs); `extra_content` pouches kept when `origin` matches the
  candidate, stripped otherwise. Gemini's converter seeds the first parallel call with Google's
  skip sentinel when signatures are gone.

### messages → `{"messages": [...]}`

Anthropic-native messages, our own walk. any-llm's openai→anthropic converter reconstructs
thinking from a completion-bridge spelling we never produce, keeps only the last signature, and
strips every key but role and content — reuse would import a signature-verification failure.

- user string → verbatim; parts: text rides, image → `{"type": "image", "source": {"type":
  "base64", "media_type", "data"}}`, file → the same shape as `document`, audio →
  `UnsupportedField` (anthropic has no audio input).
- assistant, origin match → `{"role": "assistant", "content": <native verbatim>}`.
- assistant, foreign → a text block (only when non-empty) plus one
  `{"type": "tool_use", "id", "name", "input": <arguments decoded>}` per call. Never an empty
  text block; a turn that degrades to nothing is omitted entirely. No thinking block is ever
  reconstructed.
- tool → `{"type": "tool_result", "tool_use_id", "content"}` in a user turn, consecutive results
  in one user message; empty content omits the key.
- The thinking trap: with a thinking level set, anthropic requires the final assistant turn
  before trailing tool results to start with its own signed thinking block. A foreign turn there
  cannot comply, and fabricating is forbidden, so assembly raises `UnsupportedField` naming the
  fix — a per-candidate `thinking` override. Plain-text foreign history is unaffected.

### responses → `{"input": [...]}`

- user string → `{"role": "user", "content": str}`; parts rename: text → `input_text`, image →
  `{"type": "input_image", "image_url": <data-uri str>}`, file → `{"type": "input_file",
  "file_data", "filename"}`. audio → `UnsupportedField`: responses takes no audio input, so an
  openai candidate carrying audio runs the completion dialect instead (CONFIG.md field 4).
- assistant, origin match → the native items spliced verbatim, minus any reasoning item whose
  required follow-up item is missing — a truncated history 400s otherwise.
- assistant, foreign → `{"role": "assistant", "content": str}` when text exists; one
  `{"type": "function_call", "call_id", "name", "arguments"}` per call.
- tool → `{"type": "function_call_output", "call_id", "output"}`.
- Statelessness is assembly's law, not a message spelling: every responses call carries
  `store: False` and `include: ["reasoning.encrypted_content"]`, so saved history is the only
  conversation state and reasoning replays with no server dependency. Candidate `extra` merges
  last and can override — flipping `store` on is a retention choice, not a correctness one, and
  a caller-added `include` concatenates rather than replaces (merge's list law).

## The round trip

The response exposes `message`: the history-ready assistant turn.

- completion: the readable fields as any-llm returns them, plus inline pouches, plus `origin`;
  gemini tool-call ids re-minted unique.
- messages: readable distilled from the blocks — text concat, thinking concat, tool_use →
  tool_calls — plus `origin` and `native` = the content list.
- responses: readable distilled from the items — output_text concat, summary concat,
  function_call → tool_calls (by `call_id`) — plus `origin` and `native` = the output list.

`messages + [response.message] + [tool results]` is the next call's history. Save → reload →
next call is lossless by construction: `native` stores wire-equivalent JSON, never SDK objects,
and a golden pins the save → load → send identity. `thinking_summaries` filters only the
response's readable view; `message` keeps `reasoning` regardless, or the saved view could never
be rebuilt (CONFIG.md field 7). Streams must end in this same `message` — accumulation rules
land with the stream-event spec.

## Honest limits, pinned at build

- Gemini's chat shape is any-llm's normalization, not Google's Part list: parallel text parts
  merge, and only the last signed non-call part keeps its signature. The tool + thinking loop is
  live-verified; if a seed ever shows the collapse breaking replay, the fix lands in the any-llm
  fork's converter, not in a second history format here.
- gemini ↔ vertexai serve the same models but count as different origins until a live seed
  proves signatures replay across both directions; until then that hop degrades like any foreign
  one.
- openai's web_search_call items replay the call, not its results — the results live in the
  following message item's text. A same-provider fidelity limit, not a bug.
- Build seeds: `include: reasoning.encrypted_content` on non-reasoning openai models (if gated,
  the include becomes conditional); verbatim item replay under `store: False`, ids included; the
  anthropic thinking trap's exact trigger; gemini's skip sentinel on foreign parallel calls.
- Refusals and citations are readable-view material: they ride `native` for same-origin replay
  and surface in the response-type round.
- Inline base64 is the only transport for parts. Files past the inline caps need a provider Files
  API; any-llm exposes anthropic's (upload / list / retrieve / download / delete), and no part
  references an uploaded file yet. Added when a consumer needs it.
