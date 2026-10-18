# Tools

Every tool is ours, a function tool the caller runs. No provider-hosted tools, ever. The loop
that runs them sits beside llm_master, not under it (SPEC.md). This page is the roster and
what each tool hands back.

| tool | group | result |
|---|---|---|
| read | sandbox (Modal) | text, image (image files, rendered PDF pages), PDF |
| write | sandbox | text |
| edit | sandbox | text |
| bash | sandbox | text |
| run_program | sandbox, programmatic tool calling | text |
| web_search | web (Exa) | text |
| fetch_url | web (Exa) | text |
| generate_image | generation | image |
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

## Shapes

Each tool copies a precedent the models already know, chosen once for every provider. Decided so
far: `edit` takes pi's shape, which holds up across providers; `web_search` and `fetch_url`
follow Exa's own guidance for agent use; `todo` takes the simpler of Claude Code's `TodoWrite`
and Codex's `update_plan`. The rest are settled in the tool design round.
