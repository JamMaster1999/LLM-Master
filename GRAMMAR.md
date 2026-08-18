# Config grammar

Every config field is one module: a value in, a kwargs fragment per API out.
Two rules keep the repo uniform. Inputs follow a four-step ladder. Outputs go
through one case per API. This doc is the reference for both; per-field detail
lives in CONFIG.md.

## Inputs: the ladder

Two primitives, two compositions. Use the smallest step that fits the field.

| shape | what it is | when to use it | example |
|---|---|---|---|
| scalar | a plain value | one number or string, no choices | `temperature=0.2` |
| enum | one choice from a fixed set | modes with no settings | `thinking="low"` |
| objects | scalars grouped in an object | one behavior, optional knobs; presence = on | `Search(max_searches=2)` |
| branches | enum + objects in one object | several behaviors, each configurable | `Format(type="schema", schema=Verdict)` |

Rules:

- Flat objects with orthogonal, optional keys. Never a union of shapes.
- `None` always means unset — the provider default. It is never a mode.
- Cross-field rules live in one validator: `type="schema"` needs a schema,
  `mode="none"` forbids `only`.

The sorting test: **translate branches on enums and reads everything else.**
If you branch on a variable, it was secretly an enum — promote it. If an enum
can only ever hold one value, it is ceremony — delete it.

## Outputs: one case per API

Every translate ends in the same skeleton. Every API gets its own case, in
this order, even when arms repeat.

```python
def translate(value, api: API) -> dict:
    match api:
        case "completion":
            ...
        case "messages":
            ...
        case "responses":
            ...
        case _:
            assert_never(api)
```

Rules:

- No combined cases, no `api` ternaries. Two arms that look alike are two
  facts that happen to agree today, not one law.
- `assert_never` is the extension mechanism: widen the `API` Literal and the
  type checker flags every translate that has not learned the new value.
- Logic that genuinely is one fact lives in a module-private helper that each
  case calls explicitly — `case "completion": return _openai(choice, flat=False)`.
  Helpers exist so copies cannot drift. They never hide an API from the match.
- Above the match, only input cleanup. Every answer — `{}` included — comes out
  of an API case, so each case tells that API's complete story.

## The pass-through law

Translation owns spellings; callers own content. Rename and restructure only
what differs between APIs (`parameters` → `input_schema`, nesting → flat).
Every other key rides through untouched, and the provider judges it. Never
silently drop and never invent — a format-level impossibility raises
`UnsupportedField(api, field, why)`.
