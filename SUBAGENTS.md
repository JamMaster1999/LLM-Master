# Subagents (paused)

**Status:** designed in part, not built. Paused on 2026-10-19 so the existing harness can reach
production first. This page holds what we decided, the open questions, and what we learned from
other harnesses, so the work can pick up where it stopped.

## The problem

Models are good at different work. Research can run on a cheaper model that searches longer, such
as Haiku or GPT-5.6 Luna. Desmos graphs and interactive scenes benefit from Opus. Emails and other
writing for parents may suit a different model again. We want to use the right model for each kind
of work in a way that:

- scales as models and kinds of work change,
- keeps the prompt cache mostly intact,
- and stays simple to maintain.

The old server tried this and got poor results from a lot of machinery (see Research below).

## What we decided

- **One main model per conversation.** We never switch it to route work. Caches are per model, so
  a switch makes the new model read the whole history at full price, and it never sees the old
  model's reasoning (IO.md). Switching stays available when the user asks for another model.
- **Model choice belongs to skills.** A skill may name a model in its SKILL.md frontmatter.
  - `agent(brief, skill="research")` runs a subagent on the skill's model, with the skill's text as
    its system prompt. That prompt is fixed per skill, so it caches.
  - `agent(brief)` with no skill runs a subagent on the parent's model.
  - `skill("research")` stays the override: it loads the skill into the parent's own context, with
    no subagent.
- **Only skills that define a kind of work name a model.** Format skills (docx, pptx, xlsx, pdf)
  don't, since one docx can be a parent letter and the next a worksheet. A subagent takes at most
  one skill that names a model. Format skills can sit beside it.
- **A writing skill is the first one that needs a model.**
- **The parent decides everything the subagent gets:**
  - the tools: a list, none, or all; "all" excludes the parent-only tools;
  - the brief, which the parent writes;
  - the context: a fork of the history so far, the last N messages, or nothing;
  - the skills;
  - the files: old uploads attach through their views (page images, transcripts).
- **Same client, same sandbox.** The subagent uses the parent's `LLM` facade, so it gets the rate
  limiter, retries, fallback, cost and telemetry. It also uses the parent's sandbox, sandbox lock
  and source numbering, so its writes queue behind the parent's and its citations continue the
  parent's numbers.
- **The parent gets back the final text and an agent id.** The subagent's tool calls stay out of
  the parent's history. Its transcript is kept in `Conversation`, where the sandbox can't write.
- **Continuing a finished subagent is a message to it.** Calling `agent` with the id appends the
  parent's new message to the transcript and runs the subagent again.
- **Steering is stop, then send a new message,** for the main agent and subagents alike. There's no
  cross-request message queue for now.
- **A stopped step keeps its partial text,** marked as interrupted, as Claude Code does. This
  applies to the main agent today, independent of subagents.
- **A sandbox command keeps running after a stop** until its timeout. That's accepted.

## A sketch of the first version

One tool:

```
agent(brief, skill=None, tools="all" | "none" | [names], context="fork" | N | "none",
      skills=[], files=[], continue_id=None)  ->  final text, agent id
```

1. **One loop for both.** The inner loop of `turn()` moves into
   `run(model, messages, box, max_steps)`, which `turn()` and `agent()` both call.
2. **The child's tools are a subset of the parent's toolbox.** It shares the parent's sandbox,
   lock, web client and sources. The parent-only tools are removed: `ask_user` (a child can't
   pause the user's turn), `todo` (it draws the user's plan), `deliver_file` (review and delivery
   stay in one place) and `agent` (no nesting).
3. **The child's messages are built in four steps:**
   1. The context: a fork, the last N messages, or nothing.
      - A fork answers the parent's still-open calls, including this `agent` call, with a
        placeholder, so the copy passes validation. Codex and Claude Code both do this.
      - Last N cuts at a user message, so no tool result loses its call.
   2. Then one user message holding the attached skills' text, the file views and the brief.
   3. The system prompt is the skill's text, or a short default when there's no skill.
   4. The tool list keeps the toolbox's order, so the same tools give the same bytes.
4. **Several `agent` calls in one step already run in parallel,** because the loop runs calls that
   don't change the sandbox together.
5. **Each subagent has a step limit.** The loop has none today.
6. **Stopping the turn cancels the parent and its children.** Each child's transcript keeps the
   steps it finished.

The estimate is about 200 lines on top of today's loop, plus tests.

## Open questions

1. **Can the parent keep working while a child works?**
   - `agent` would return the id at once, and the child would run as its own task in the same
     container.
   - Its report would land in the parent's history at the parent's next step.
   - If the parent answers while children are still running, the turn waits for the next report
     instead of ending.

   This is Claude Code's background agents, about 40 lines more. Sina wants the core to allow it.
2. **Can the user stop one child and let the rest run?**
   - The browser can't reach into a running turn, so this needs a small separate request.
   - That means a `/stop/{agent_id}` endpoint that skips the conversation lock, a flag in a Modal
     Dict keyed by agent id, and a watcher that cancels the child.

   It's about 30 lines, and the first piece that crosses requests and containers.
3. **Can the user send a message to one running child?** It's the same channel as question 2,
   carrying text instead of a flag. A Modal `Queue` with one partition per agent id fits: it is
   shared across containers, holds items up to 1 MiB, and clears 24 hours after its last item.
   About 10 lines once question 2 exists.
4. **How does the UI show what a child is doing?**
   - After the fact is easy: render its saved transcript.
   - While it runs, the child's chunks would go into the parent's stream, tagged with the agent id.
     That needs one loop change, letting a tool emit chunks while it runs.
5. **Should children ever outlive the parent's request?** Closing the main stream stops everything,
   because the children live inside that request. Outliving it would need a Modal function call
   per child and a shared transcript store. That's the hard version, and we expect to leave it
   out.
6. **What does a skill's `model` field hold?** A model id, or a list of candidates for fallback?
   Ids in each SKILL.md, or names resolved through one table?
7. **What's the default step limit, and can a skill set its own?**
8. **Is the parent-only list right?** It holds `ask_user`, `todo`, `deliver_file` and `agent`. The
   widget tool doesn't exist yet.
9. **How does a child's report enter the parent's history?** As a user-role message marked as
   subagent output, with instructions inside it carrying no authority, as Claude Code does?
10. **When is a fork to another model worth it?** That model reads the whole history cold and
    without the parent's reasoning.
11. **Does each model-bound skill beat the parent model with the skill loaded?** Measure before
    shipping each one: teacher preference, cost, cache reads and latency. pi's author argues
    subagents transfer context poorly and act as "a black box within a black box."

## Research

### What models did differently in the scenario runs

These come from six rounds of the PowerPoint scenario on four models (PR #28).

| work | what we saw |
|---|---|
| research | GPT-6.1 Sol and Gemini 4 Argon were the most thorough. Sonnet 5.5 was light but accurate. Gemini 3.8 Flash skipped research until the prompt required it |
| slide writing | Sol wrote the leanest, clearest decks, and Sonnet's were clean and short. Both Geminis added the most teaching scaffolds but crammed 100+ words onto a slide |
| code that builds files | All four built working python-pptx. Sonnet finished in 2 to 4 minutes. The Geminis needed 10 to 26 check-and-fix cycles |
| review | Sol and Argon read pages at full size and fixed what they saw. Sonnet judged from thumbnails |
| wrap-up | Sol stayed under 200 words. Flash ran to 450 |

The best decks came from one model writing, checking and fixing in a single loop. Splitting the
writing from the review hands one model's errors to a model that didn't make them.

### The old server

`uflo-AI-server` (`chat/`, `question/` and its own `llm_master/`) shows what to avoid:

- **A classifier ran in front of every message.** It was a 338-line prompt with no caching, and it
  still ran after the user had picked manually.
- **The model could change on every turn, and each feature's prompt went into the last user
  message.** Both defeated caching.
- **History was rebuilt as plain text.** Tool results were stored as XML tags and found again by
  regex. Native tool calls, thinking and Gemini's thought signatures were lost.
- **Prompt and model choices lived in an external database** behind hard-coded prompt ids, fetched
  on every request with no timeout.
- **Model names were never checked at startup,** so broken names failed with no fallback.
- **Cost tracking counted only the final call,** and the usage math hid cache hits.

What worked: a simple config mapping each tutor mode to a model, which made vendor A/B tests easy.
Claude cache breakpoints in the roleplay path also worked.

### Routers and caching

- **[OpenRouter's Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router)**
  classifies each prompt into one of about 30 task types. It then picks the model with the most
  community spend on that task over the last week. Once it sees a cache hit, it pins the model and
  provider for the conversation to keep the cache warm. The router itself gives up on switching
  models mid-conversation.
- **[RouteLLM](https://github.com/lm-sys/RouteLLM) and
  [Not Diamond](https://docs.notdiamond.ai/reference/token_model_select_v2_modelrouter_modelselect_post)**
  pick a strong or cheap model per request from trained routers. That costs an extra call per
  request and needs evaluation data to calibrate.
- **[LiteLLM](https://docs.litellm.ai/docs/auto_router/prompt_caching.md)** measured that a switch
  isn't an eviction: on 4,684 switch-backs, a model's cache was 97.4% warm at a 5-minute TTL. The
  model that's switched to still reads everything since its last call at full price.
- **[Anthropic's caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)**
  needs a byte-exact prefix. It invalidates in the order tools, then system, then messages. The
  default TTL is 5 minutes, and a 1-hour option costs more.

### Delegation patterns

- **[OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents/orchestration):**
  handoffs pass control of the conversation to another agent. Agents as tools keep the manager in
  control.
- **[Claude Code skills](https://code.claude.com/docs/en/skills)** can set `model` and
  `context: fork`, which runs the skill in its own subagent.
- **[Anthropic's advisor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool.md)**
  (beta) lets a fast executor consult a stronger model mid-generation. It's provider-hosted and
  Claude-only, which rules it out for us.

### pi, Codex and Claude Code

| | [pi](https://github.com/earendil-works/pi) (MIT) | [Codex](https://github.com/openai/codex) (Apache-2.0) | Claude Code (proprietary; read for design only) |
|---|---|---|---|
| subagents | not in core; an example extension | built in | built in |
| first version | about 170 lines of logic (Dec 2025) | about 600 lines of Rust, 4 tools, built Jan 6 to 12, 2026 | no history in the mirror |
| today | a 1,036-line example, plus an experimental `durable` package | about 10,750 lines | about 1,500 core, 6,800 in all |
| roles | markdown with `name, description, tools, model` | TOML layered over the parent's config | markdown frontmatter: model, tools, skills, maxTurns and more |
| context | the brief only | the brief, or a full fork that keeps text and drops tool calls and reasoning | the brief, or a full fork on the same model, sharing its cache |
| tools | set per role | the parent's set | set per agent type, with parent-only tools denied |
| messaging | durable: spawn, send, stop, status | spawn_agent, send_input, wait_agent, close_agent | background agents with notifications; SendMessage |
| nesting | blocked in durable | depth 1 by default | blocked; the docs now allow 3 levels |
| step limit | none | none | `maxTurns` |

**pi** leaves subagents out on purpose. Its author argues context transfers poorly and that needing
context mid-session means the work wasn't planned. The first subagent example spawned a fresh `pi`
process with a role prompt, a tool allowlist and the brief, and returned only its final text.
`durable` adds two-way messaging through two loop hooks: steering messages, read at the next step,
and follow-up messages, read after an answer. A child's answer reaches the parent as a follow-up
message.

**Codex** grew in small steps:

| date (2026) | added |
|---|---|
| Jan 6 to 12 | spawn, send input, wait, close |
| Jan 15 | a role table mapping each role to instructions and a model |
| Jan 26 | a depth limit |
| Feb 7 | resume |
| Feb 19 | completion notifications |
| Feb 26 | forks |
| Mar 31 | a last-N-turns fork |
| Oct 6 | the last-N-turns fork removed, with no reason given |

`spawn_agent` returns an id at once, the child runs in the background, and `wait_agent` returns
each agent's status with its last message. A first fork answers the open `spawn_agent` call inside
the child's copy and tells the child to treat the next message as its task.

**Claude Code** defines an agent as markdown: frontmatter for settings, the body as its system
prompt. Its fork copies the parent's system prompt, tools and history, answers every open call with
the same placeholder, and tells the child not to spawn. It adds preloaded skills as a hidden user
message. It returns only the child's last text. Background agents report through a notification
read at the parent's next tool step. A message to a running agent is read at its next step, and a
message to a finished one resumes it from its saved transcript.

### What all three agree on

- A role table or agent file with a prompt and a model.
- One tool to start a subagent.
- Only the final text goes back to the parent.
- Nesting is blocked by default.
- The tools that need the user (asking questions, approvals) stay with the parent.
