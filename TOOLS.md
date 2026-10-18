# Tools

Every tool is ours, a function tool the caller runs. No provider-hosted tools, ever. The loop
that runs them sits beside llm_master, not under it (SPEC.md). This page is the roster, what
each tool takes, and what it hands back. The code lives in `agent/`.

| tool | group | result |
|---|---|---|
| read | sandbox (Modal) | text, image (image files, rendered document pages), PDF |
| write | sandbox | text |
| edit | sandbox | text |
| bash | sandbox | text |
| run_program | sandbox, programmatic tool calling | text |
| web_search | web (Exa) | text |
| fetch_url | web (Exa) | text |
| generate_image | generation | text and image |
| generate_audio | generation | text (the audio file goes to the user) |
| file_search | knowledge | text |
| search_uflo_docs | knowledge: Uflo's own product docs | text |
| mcp | knowledge: ours (student submissions, grades, manual regrades, regrade runs, quiz export to the LMS), and an LMS through LTI | text |
| deliver_file | delivery | text |
| widget | delivery: one tool, many skills | text |
| todo | loop | text |
| ask_user | loop | text |

`generate_image` makes an image with llm_master's `image()`, or edits one: the parent image and
the new instruction go out together, and a small record beside each image names its parent.
`generate_audio` is one-shot speech through `speech()`. Both run openai first with gemini as the
fallback.

Widgets and citations are text. A result is text, an image, or a PDF; nothing else. No API takes
audio inside a tool result, so audio never comes back as one.

## Running tools

The caller holds one `Toolbox` per conversation. It lists the tools for the config and runs the
model's calls:

```python
from agent import Toolbox

def lookup_grade(student: str) -> str:
    """The student's current grade in this course."""
    ...

box = Toolbox("conv_8f2", tools=[lookup_grade])
response = await llm.query(..., config=Config(tools=box.specs))
for call in response.message.calls:
    result = await box.run(call["name"], call["arguments"])
    history.append({"role": "tool", "call_id": call["id"], "content": result.content})
```

- `box.specs` lists the built-in tools and the caller's own, as openai function dicts.
- A custom tool is a plain Python function, sync or async. Its schema comes from its signature
  and docstring through any-llm's `callable_to_tool`, as in CONFIG.md field 6. It returns a
  string, and runs through the same `run` as every built-in.
- `run` returns a `Result` with three fields:
  - `content`: the tool message's content, a string or text, image and file parts (IO.md).
  - `files`: files for the user, from `deliver_file` and the `generate_*` pair.
  - `sources`: the numbered sources, `{n, url, title}`, so the backend can link `[n]` in the UI.
- Source numbers continue across the conversation. A backend that rebuilds the toolbox for an
  existing conversation passes the count so far: `Toolbox("conv_8f2", sources=12)`.
- `ask_user` waits for the user, so the loop ends the turn before running it (Shapes below).

## Shapes

Each tool copies a precedent the models already know, chosen once for every provider. The
tables give each parameter; the JSON schemas in `agent/` are the exact spelling.

Rules every tool shares:

- **Paths** are relative to `/workspace`, the conversation's folder in its sandbox, or absolute.
  Only `/workspace` survives the sandbox stopping.
- **Errors** come back as the result's text. IO.md's tool message has no error flag, and the
  model reads the message and recovers.
- **Long text** is cut at 2000 lines or 50 KB, whichever comes first, with a notice that says
  how to read on (pi's limits).
- **Citations** are numbers. `web_search`, `fetch_url`, `file_search` and `search_uflo_docs`
  number their sources, one count across the conversation, and the model writes `[3]` where it
  uses source 3.

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
| image (png, jpg, gif, webp; bmp, tiff and heic convert to png) | `Read image file [image/png]` and the image, scaled to at most 2000 px on its long side |
| PDF | `Read PDF file report.pdf [pages 1-10 of 42]` and the PDF itself, cut to at most 10 pages |
| document (docx, doc, odt, rtf, pptx, ppt, odp) | `Rendered deck.pptx [pages 1-6 of 6]` and one image per page: LibreOffice → PDF → pdftoppm. This is how the model checks the files it builds |
| spreadsheet (xlsx, xls, ods, csv, tsv) | every sheet as CSV under a `## <sheet>` heading, when it all fits in 25 KB. Larger: the first 20 rows of each sheet, its size, and a note that the full data stays in the sandbox for `bash` or `run_program` |
| audio, video | an error: read can't return them. User audio up to 10 minutes already reached the model in the user message |

A missing file returns `File not found: <path>`; a directory or a binary file returns an error
that points at `bash`.

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
show up in practice. The model sees
`Successfully replaced 2 block(s) in <path>.` and no diff. Failures use pi's wording:

- `Could not find edits[1] in <path>. The oldText must match exactly including all whitespace and newlines.`
- `Found 3 occurrences of edits[0] in <path>. Each oldText must be unique. Please provide more context to make it unique.`
- `edits[0] and edits[1] overlap in <path>. Merge them into one edit or target disjoint regions.`
- `No changes made to <path>. The replacements produced identical content.`

pi repairs three input mistakes before validating, and so do we: `edits` sent as a JSON string,
one edit object without the array, and pi's older top-level `oldText`/`newText`.

### bash

pi's `bash`, with Claude Code's default and ceiling on the timeout, because a hosted sandbox
can't wait forever.

| parameter | type | meaning |
|---|---|---|
| command | string, required | runs with `bash -c` in `/workspace`, a fresh shell each call |
| timeout | number | seconds, default 120, at most 600 |

The result is stdout and stderr interleaved, cut from the top so the end survives. When cut, the
full output is saved in the sandbox and the notice names it:
`[Showing lines 3001-5000 of 5000. Full output: /tmp/bash-1f3a.log]`. Empty output reads
`(no output)`. A failure appends `Command exited with code 2` or
`Command timed out after 120 seconds`.

### run_program

Anthropic's programmatic tool calling, run in our sandbox. The model writes Python that calls
the other tools, and only what the program prints comes back, so intermediate results never
fill the context.

| parameter | type | meaning |
|---|---|---|
| code | string, required | a Python program, run in `/workspace` |
| timeout | number | seconds, default 120, at most 600 |

Inside the program every other tool except `ask_user` is an async function, as in Anthropic's
version: it takes one dict of arguments and returns the result's text. Top-level `await` works,
and `asyncio.gather` runs calls in parallel:

```python
pages = await asyncio.gather(*[web_search({"query": q, "objective": "..."}) for q in queries])
print(max(pages, key=len)[:2000])
```

The result reads like `bash`: output interleaved, cut from the top, and
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
as Exa asks. No results reads `No search results found.`

### fetch_url

Exa's `get_contents` tool, under our roster name.

| parameter | type | meaning |
|---|---|---|
| urls | array of strings, required | absolute URLs; several are read in one call |

Each page comes back numbered like a search result, with its text cut at 10,000 characters
(exa-py's default) in place of highlights. A page that fails reads
`[5] Error fetching <url>: <reason>`, and the others still return.

### generate_image

| parameter | type | meaning |
|---|---|---|
| prompt | string, required | what to make, or how to change the image |
| image | string | the path of an image to edit; omitted makes a new one |

An edit sends the parent image and the prompt as one request (SPEC.md, Media). The new image is
saved as `images/<id>.png`, with `images/<id>.json` beside it holding the parent's path (or
null), the prompt, and the candidate that made it. The model gets `Saved images/<id>.png` and the
image; the user gets the image. No size parameter: `image()` takes none, and the prompt can ask
for a shape.

### generate_audio

| parameter | type | meaning |
|---|---|---|
| text | string, required | exactly what to say |

The speech is saved as `audio/<id>.mp3` and goes to the user. The model gets
`Saved audio/<id>.mp3 and sent it to the user.`

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
| plan[].status | `pending`, `in_progress` or `completed` | at most one step is `in_progress` |

Returns `Plan updated`. The tool keeps no state: the UI draws the list from the latest call's
arguments, and the model sees its own last call in the history.

### ask_user

Claude Code's `AskUserQuestion`, without its terminal-only `preview` field.

| parameter | type | meaning |
|---|---|---|
| questions | array of 1–4, required | the questions, shown together |
| questions[].question | string | the full question, ending with a question mark |
| questions[].header | string | a chip label of at most 12 characters |
| questions[].options | array of 2–4 | the choices; the UI adds a free-text "Other" itself |
| questions[].options[].label | string | 1–5 words |
| questions[].options[].description | string | what the choice means |
| questions[].multiSelect | boolean | whether several options may be picked |

The user answers this tool, not the toolbox. When the model calls it, the loop ends the turn
and the browser shows the questions (AI SDK's client-side tools). The answers come back with the
next request, and the loop runs the call with them added, as Claude Code's permission component
does: `box.run("ask_user", {**arguments, "answers": {question: answer}})`. Multi-select answers
are joined by commas. The model then reads Claude Code's text:

`User has answered your questions: "Which grade level?"="Grade 9". You can now continue with the user's answers in mind.`

No answers reads `User declined to answer questions`.

### file_search

OpenAI's `file_search`. Its call carries a list of queries, the shape OpenAI's models emit.

| parameter | type | meaning |
|---|---|---|
| queries | array of strings, required | one or more searches, run together |

It searches the conversation's ingested files (Files below). The backend fixes the scope and the
result count; the model only writes queries. Each passage comes back numbered, like a search
result, with the file's name and path in place of a URL:

```
[7] File: unit-3-notes.pdf (uploads/unit-3-notes.pdf), page 12
The mitochondria...
```

### search_uflo_docs

| parameter | type | meaning |
|---|---|---|
| query | string, required | a question about Uflo, in plain words |

Searches Uflo's own product docs (docs.uflo.io). Results come back numbered like `web_search`,
with excerpts, so the model cites the docs it used.

### mcp

One tool for every MCP server we connect, so the tool list stays fixed for the prompt cache
while servers and their tools change per school.

| parameter | type | meaning |
|---|---|---|
| server | `uflo` or `lms` | ours (submissions, grades, regrades, quiz export) or the school's LMS through LTI |
| tool | string | the server's tool to call; omitted lists the server's tools with their input schemas |
| arguments | object | the tool's arguments |

The result is the MCP tool's text. Skills that read through our server (analytics, rubric writer)
name the tools they use, so the model rarely needs the listing.

### widget

One tool for every widget. The skills teach each widget's component; the tool only shows it.

| parameter | type | meaning |
|---|---|---|
| code | string, required | the widget in OpenUI Lang, built from our component library |

The backend parses the code against the library and sends it to the browser as a `data-widget`
part. The model gets `Shown to the user.` or the parser's errors, so it can fix the code.

## Files

A user's upload lands in the sandbox under `uploads/`, so every tool can reach it.

- A PDF up to 10 pages and audio up to 10 minutes also go into the user message natively.
- Longer ones go through ingestion: indexed for `file_search`, plus a readable text file beside
  the original (`uploads/report.pdf.txt`) that `read` pages through.
- A spreadsheet up to 25 KB goes into the user message as CSV. A larger one stays in the sandbox,
  and the user message carries a preview.
- Video is not supported.
