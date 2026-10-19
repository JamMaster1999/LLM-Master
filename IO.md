# Messages, responses and streams

This page defines the message format llm_master accepts, how each provider API receives it, the
response a call returns, and the chunks a stream yields. Config fields are in CONFIG.md, code
conventions in GRAMMAR.md, and tools in TOOLS.md.

The caller keeps a conversation as a list of messages in llm_master's format. On every call,
llm_master converts that list into the format of the API it calls: Anthropic Messages, OpenAI
Responses, or chat completions for Gemini, Vertex AI and Fireworks.

llm_master uses its own format instead of OpenAI's chat format for two reasons:

- A chat tool message holds only text, so there's nowhere to put an image or PDF a tool returns.
- An assistant message needs fields the chat format lacks: the reasoning text, which model wrote
  it, and the provider's raw output for replay.

Two terms come up throughout:

- **candidate** — a provider and model a call can go to. A call lists its candidates in order:
  the first is the primary, the rest are fallbacks, tried one by one if the ones before fail.
- **native** — a provider's raw output for one assistant message, stored so it can be sent back
  to the same model unchanged.

## Roles

| role | fields | meaning |
|---|---|---|
| `user` | `content` | what you send |
| `assistant` | `content`, `calls`, `reasoning`, `origin`, `native` | what a model said |
| `tool` | `call_id`, `content` | one tool call's result |

- There's no system role. The system prompt is the `system` config field (CONFIG.md field 1), and
  a message with role `system` fails validation.
- `content` per role:
  - user: a string or a list of parts (text, image, file, audio).
  - tool: a string or a list of parts (text, image, file).
  - assistant: a string, empty (`""`) when the message only calls tools.
- `calls`: a list of tool calls, each `{"id": str, "name": str, "arguments": dict, "wire_id":
  str}`. `arguments` holds JSON values only; a caller-written call has no `wire_id`.
  - `id` is ours; `wire_id` is the id the provider gave the call (see Ids).
- One `Message` class covers all three roles.
  - `role` decides which fields apply, and one validator checks the rules between fields.
  - Plain dicts validate through the same class; the class itself is there for editor
    autocomplete.
  - Unknown keys fail validation, so a typo is caught.
  - Save a message with `message.model_dump(mode="json", exclude_none=True)` and load it with
    `Message.model_validate`; the reloaded message builds the same request.

## Parts

Parts are a fixed set, told apart by `type`. An unknown `type` fails validation.

- text — `{"type": "text", "text": str}`
- image — `{"type": "image", "data": <base64>, "mime": str}`
- file (PDF) — `{"type": "file", "data": <base64>, "mime": "application/pdf", "filename": str}`
- audio — `{"type": "audio", "data": <base64>, "format": str}`, user messages only

Messages hold base64 data only, never a URL or a file path. `Image`, `File` and `Audio` take a
data-uri, a URL or a local path and encode it once, when the part is created, so every later call
sends the same bytes (see The same-bytes rule). HEIC becomes JPEG. A source that can't be read, an
`Image` that isn't an image, or a `File` that isn't a PDF raises `PartError` before any model is
billed. Never pass a user's input as a source: a path or URL is read as given.

Size limits are left to the provider. For reference: Anthropic's API rejects an image over 10 MB
(5 MB on Bedrock and Vertex AI), and Gemini takes up to 100 MB of inline data per request, 50 MB
for a PDF.

## Assistant messages: readable fields and native

An assistant message holds two views of the same answer.

The readable fields are `content`, `calls` and `reasoning`. Our code reads these, a frontend
renders them after a reload, and they are all another model ever sees of the message.
`reasoning` is the thought text the provider exposes: Anthropic's thinking, OpenAI's summaries,
Gemini's thoughts. They differ in how much they reveal but display the same way. `reasoning` is
never sent to a provider.

A message a provider produced has two more fields:

- `origin` — `{"provider": ..., "model": ...}`, the candidate that produced it. It's set from the
  candidate, not from the model name in the response, because Gemini's response says `google/genai`.
- `native` — the provider's raw output as JSON, never re-validated:
  - Anthropic: the response's `content` list (signed thinking, redacted thinking, text,
    tool_use, and any block type added later).
  - OpenAI: the response's `output` list (reasoning with `encrypted_content`, message,
    function_call, and so on).
  - Gemini, Vertex AI, Fireworks: the assistant message as any-llm returned it, including
    Gemini's thought signatures in `extra_content` and Fireworks' `reasoning_content`.

An assistant message the caller wrote has no `origin` and no `native`.

**The replay rule.** When a candidate matches a message's `origin` (same provider and same
model), llm_master sends `native` unchanged and ignores the readable fields. Every other candidate
gets only `content` and `calls` (see Messages from other models). Reasoning never goes to a
different model. Within one provider, the servers silently drop reasoning a sibling model can't read, so a hop such as gpt → gpt-mini or opus → sonnet loses the reasoning without an error. Stored history is never
changed, so a later call back to the original model replays it in full.

## Ids

Each tool call gets our own id when its message enters history: `lm_` plus 24 hex characters.
The id is stored and never regenerated. The provider's own id is kept as `wire_id`. It's used
only when sending `native` back to the model that made the call; in that case the tool result
points at `wire_id`, and everywhere else at `id`.

Why our own id: Anthropic accepts only letters, digits, `_` and `-`; OpenAI Responses caps
`call_id` at 64 characters; Gemini 3 matches results to calls by id. One id that meets all three,
created once, never depends on what another provider generated. Ids in caller-written calls must
match `^[A-Za-z0-9_-]{1,64}$`.

`query()` never runs a tool; the caller runs them (CONFIG.md field 6).

## The same-bytes rule

Building a request from history depends only on the history and the candidate. It does no I/O
and uses no clock and no randomness.

The reason is prompt caching. Every provider's cache only hits when the start of a request is
byte-for-byte the same as an earlier one. If one earlier byte changes (a re-encoded image, a
regenerated id, a reordered field), the cache misses from that point on, and every later call
pays full price for the whole conversation.

So ids are created once, parts hold base64 from the moment they're created, `native` stays raw
JSON, and golden tests (saved expected requests) pin every request shape below. Changing one
busts the cache for every saved conversation.

## Validation

One validator checks every rule, before anything is sent:

- **The list**
  - It can't be empty.
  - Roles are user, assistant and tool. A system-role message fails.
- **Fields per role**
  - `call_id` is required on tool messages and forbidden on the others.
  - `calls`, `reasoning`, `origin` and `native` belong to assistant messages only.
  - `native` requires `origin`.
- **Content**
  - Audio parts appear only in user messages.
  - Tool messages hold text, image and file parts.
  - An assistant message with nothing in it (empty content, no calls, no native) fails.
- **Calls**
  - Every call's `arguments` is a JSON object.
  - Call ids are unique across the conversation.
- **Tool loops** — one assistant message can make several calls at once (parallel calls).
  - Each call gets its own tool message, and they all come right after the assistant message.
  - Each `call_id` matches one of that message's calls, and none repeats.
  - Every call is answered before the next user or assistant message.
  - Anthropic, OpenAI and Gemini each return a 400 on a different one of these mistakes; we
    reject all of them up front.

## Building the request, per API

io/messages.py has one case per API, following GRAMMAR.md. The candidate's provider and model
decide which messages replay `native`. The system prompt is passed in because the completion API
has no separate parameter for it.

### completion → `{"messages": [...]}`

any-llm's converters expect OpenAI's chat format, so this case writes it.

- `system` set: one system-role message goes first.
- user: string content as is. Parts are renamed:
  - image → `image_url` with a data-uri
  - file → `file` with `file_data`
  - audio → `input_audio`. An OpenAI candidate with audio uses this API (CONFIG.md field 2).
- assistant from the same model: `native` unchanged, including Gemini's signatures and
  Fireworks' `reasoning_content`.
- tool: `{"role": "tool", "tool_call_id", "content"}`, parts renamed as for user.
  - Gemini 3 receives image and PDF parts as `functionResponse.parts`.

### messages → `{"messages": [...]}`

Anthropic's Messages format, built by our own code. any-llm's converter is not reused: it drops
every key except role and content, and rebuilds thinking from a format we never produce.

- user: a string unchanged. Parts become blocks:
  - text → as is
  - image → `{"type": "image", "source": {"type": "base64", "media_type", "data"}}`
  - file → the same shape as a `document` block
  - audio → `UnsupportedField` (Anthropic has no audio input)
- assistant from the same model: `{"role": "assistant", "content": <native unchanged>}`, or
  nothing when `native` is empty.
- tool: `{"type": "tool_result", "tool_use_id", "content"}` inside a user message.
  - Consecutive results go in one user message.
  - Parts become text, image and document blocks.
  - Empty content leaves out the `content` key.

### responses → `{"input_data": [...]}`

- user: a string → `{"role": "user", "content": str}`. Parts are renamed:
  - text → `input_text`
  - image → `{"type": "input_image", "image_url": <data-uri>}`
  - file → `{"type": "input_file", "file_data", "filename"}`
  - audio → `UnsupportedField`. Responses takes no audio input, so an OpenAI candidate with
    audio uses the completion API instead (CONFIG.md field 2).
- assistant from the same model: the `native` items inserted unchanged.
  - Except a reasoning item whose required next item is missing: that one is dropped, because a
    truncated history returns a 400 otherwise.
- tool: `{"type": "function_call_output", "call_id", "output"}`, where output is a string or a
  list of `input_text`, `input_image` and `input_file`.
- Every Responses call sends `store: False` and `include: ["reasoning.encrypted_content"]`.
  - Saved history is then the only conversation state, and reasoning replays without depending
    on OpenAI's servers.
  - A candidate's `extra` is merged last and can override these. Turning `store` on is a
    data-retention choice, not a correctness one.
  - An `include` the caller adds is appended to ours, not swapped in.

## Messages from other models

An assistant message from another model, or one the caller wrote, is sent as its text and its
calls only, with our ids, in the candidate's own format. No reasoning, no signatures, no provider
ids. Its tool results are sent exactly as they would be after the candidate's own calls.

- completion: `{"role": "assistant", "content": <text or None>, "tool_calls": [{"id", "type":
  "function", "function": {"name", "arguments": <json str>}}]}`, with `tool_calls` only when
  there are calls. A message with neither text nor calls is left out.
  - Gemini: any-llm adds Google's documented `skip_thought_signature_validator` to the first call
    without a signature, so the call is accepted.
  - Fireworks: nothing more is needed. On deepseek-v4p1-flash, kimi-k3 and glm-5p3, a message
    without `reasoning_content` was accepted and answered (live-tested).
- messages:
  - a text block, only when the text isn't blank (Anthropic returns a 400 on a blank one);
  - then one `{"type": "tool_use", "id", "name", "input": <arguments>}` per call;
  - a message that ends up empty is left out.
- responses:
  - `{"role": "assistant", "content": str}` when there is text;
  - then one `{"type": "function_call", "call_id", "name", "arguments": <json str>}` per call,
    without an item `id`. A function call with an `fc_` id but without its reasoning item
    returns a 400.

Switching models in the middle of a tool loop needs nothing extra. Claude in adaptive mode turns
thinking off for that one request instead of failing
([thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)). Gemini accepts the
skip value, and Fireworks accepts the plain message.

## The response

A call returns a `Response`. Its `message` is the assistant message, ready to append to history:
readable fields extracted from the provider's output, our ids created, `origin` set, `native`
attached.

- completion: content, tool calls and reasoning as any-llm returns them; `native` is that
  message.
- messages: text joined, thinking joined, each tool_use becomes a call; `native` is the content
  list.
- responses: output_text joined, summaries joined, each function_call becomes a call (matched by
  `call_id`); `native` is the output list.

A tool call the model didn't finish, cut off by `max_tokens` or a refusal, can't run, so the call
raises a typed `CutCall` instead of returning it. An OpenAI response with status `failed` raises a
`ProviderError` when a retry might help, else an `InvalidRequestError` (SPEC.md Failures).

`messages + [response.message] + [tool results]` is the next call's history. Saving, reloading
and calling again loses nothing, because `native` is stored as raw JSON rather than SDK objects;
a golden test pins save → load → send.

Text, calls and reasoning live only on `message`; nothing is copied to the top level. Whether a
user sees `reasoning` is the caller's choice when rendering. The other fields describe this one
call and never go into history:

- `parsed` — the `response_format` object when the model stopped normally: an instance for a
  schema class, the parsed JSON for a schema dict or `type="json"`. None otherwise, and when the
  text isn't JSON or a schema class rejects it; the text stays in `message.content` for the caller
  to judge. A schema dict is the provider's to enforce: its JSON comes back unchecked.
- `finish` — why the model stopped, in any-llm's words: `stop`, `length`, `tool_calls` or
  `content_filter`. A model's own refusal reads as `content_filter` too; its text, if any, is in
  `message.content`. Errors raise a typed exception, so `finish` never says error.
- `usage` — input, output, reasoning, cache-read and cache-write tokens. `input` counts every
  prompt token, cache reads and writes included; `output` counts reasoning. Anthropic's 1-hour
  cache writes and audio and image tokens are broken out too, because they're priced apart. None
  when the provider reported no usage: unknown, never zero.
- `cost` — dollars, or None when the usage or the model's price is unknown (SPEC.md Money).
- `attempts` — every candidate tried, in order, with the one that answered last. One entry when
  the first candidate answers. Each attempt holds:
  - provider, model and latency;
  - `errors`: one typed exception per failed try, in order, empty when the first try worked.
    Each carries any-llm's `message`, `status_code`, `code` and `error_type`, plus the SDK's
    original exception with the provider's request id. The answering attempt can have errors
    too, when a retry succeeded.
- `latency` — wall time of the whole call after the moderation pre-check, including waits between
  retries and for the rate limiter, so it can exceed the sum of the attempts.

Timestamps, message ids and user feedback describe the conversation, not the call. The caller
stores them next to the message, never inside it.

Generated images and audio never appear on a `query()` response; they come from `image()` and
`speech()` (SPEC.md, Media).

## Streams

A stream yields chunks in the AI SDK UI message stream protocol, in exactly the shape the
browser's `useChat` reads ([stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol)).
The caller can forward them unchanged. When the loop ends, `stream.response` holds the same
`Response` a normal call returns. The stream itself never yields it, so `native` can't reach the
browser by mistake.

```python
stream = await llm.query(..., stream=True)
async for chunk in stream:
    await sse.send(chunk)
history.append(stream.response.message)
```

For each model call, llm_master yields:

- `start-step` when the attempt's first event arrives, `finish-step` once its `Response` is built:
  a request the provider rejects yields neither.
- `reasoning-start` / `-delta` / `-end`: the thought text, always. The caller forwards or drops
  it.
- `text-start` / `-delta` / `-end`: the answer, a refusal's text included.
- `tool-input-start` / `-delta` / `-available`: each call's name, then its arguments as they
  arrive, then the complete arguments. A call cut off mid-arguments never reaches `-available`:
  the step resets, and the next candidate streams from the start, or the stream raises `CutCall`
  when none is left. `toolCallId` is our `lm_` id, created when the
  call starts streaming, so it matches `response.message.calls`.
- `reset-step` when an attempt fails mid-stream. The browser removes that attempt's output, and
  the next attempt (a retry or the next candidate) streams from the start under a new
  `start-step`. Earlier steps stay.

The caller sends everything else: the message-level `start` and `finish`, tool outputs, files
and `data-*` widgets. When every candidate fails, the stream raises the same typed error a normal
call raises; llm_master never sends an `error` chunk. A stream cut off before it finishes is a
typed `ProviderError`: its step resets and the try runs again (SPEC.md Failures). A half-streamed
tool call can't be run: tools run from `response.message.calls` after the loop ends.

To abort, leave the loop or cancel the task that iterates it: the provider's stream closes.
`await stream.aclose()` closes it at once, from the iterating task.

The final `message` is built from the stream with the same extraction as a normal call, and
`native` from the provider's own stream events. Usage comes from the last usage event, never a
sum, because Gemini sends a running total on every chunk.

## Known limits

- Gemini's `native` is any-llm's simplified message, not Google's original list of parts.
  - Text before and after a call is joined, and only the last text signature is kept.
  - No error: Google requires signatures only on calls, and those are kept.
  - The cost may be slightly weaker reasoning on later turns about the in-between progress text.
  - Gemini's implicit cache is unaffected (live-tested: a follow-up read 22.6k of 24.7k prompt
    tokens from cache across the rebuilt messages, the same as an identical repeat).
- any-llm 1.37.0 drops Gemini's call ids, and Gemini then matches results to calls by position.
  - Results sent in a different order from the calls come back swapped, with a 200
    (live-tested).
  - Fork PR #18 sends ids in both directions.
  - Until it and PRs #17, #20, #21 and #22 are released upstream, llm_master installs any-llm from
    a fork build that includes them.
- Gemini and Vertex AI serve the same models but count as different origins, because AI Studio
  and Vertex reject each other's thought signatures.
- Parts travel only as inline base64. Files above the inline limits would need a provider's Files
  API. any-llm exposes Anthropic's and OpenAI's (upload, list, retrieve, download, delete) and
  Gemini's (no download), but no part can reference an uploaded file yet. That gets added when
  someone needs it.
