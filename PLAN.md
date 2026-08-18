# llm_master v2

Clean-slate rewrite: a lite wrapper around [any-llm](https://github.com/mozilla-ai/any-llm) (our fork's main: upstream + carried fixes, until upstream releases them).
Old `main` stays frozen for existing consumers. This branch (`v2`) replaces everything.

## Plan

1. **Fork** — this repo, `v2` branch. Done.
2. **PR-vs-build split** — done. Four upstream PRs; everything else lives in the wrapper.
3. **Upstream PRs** — file issues first for the two gemini ones, per any-llm's contributing rules.
   - Gemini reasoning bundle: `reasoning_effort` mapping + thoughts tokens in usage (issue first)
   - Gemini image generation, nano-banana included (issue first)
   - Anthropic `tool_choice` default fix (straight to PR)
   - Gemini `grounding_metadata` + `inline_data` surfacing (straight to PR)
4. **API spec, then wrapper** — write the API as a doc first: signatures, fallback candidate lists, config dict, response types. Implement after review.
5. **Tests** — unit + live API tests. Two benchmark gates: conversion overhead at concurrency-50, streaming time-to-first-token.

Consumer migration and the merge to `main` come later. Out of scope for now.

## Reference

Feature map (what any-llm covers natively vs. what we build): https://claude.ai/code/artifact/082d9368-29ff-46fe-a2c0-9b09033998a3
