"""The toolbox's own behaviour, for the tools that need neither Modal nor Exa, and its schemas at each provider."""
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from agent import Toolbox
from llm_master.types import Config, Model

SKILLS = Path(__file__).parent / "skills"

PLAN = [{"step": "Draft the quiz", "status": "in_progress"}, {"step": "Write the key", "status": "pending"}]
QUESTIONS = [{"question": "Which grade level?", "header": "Grade", "multiSelect": False,
              "options": [{"label": "Grade 9", "description": "Ninth grade"},
                          {"label": "Grade 10", "description": "Tenth grade"}]}]


async def test_todo_acknowledges_the_plan():
    result = await Toolbox("c").run("todo", {"plan": PLAN})
    assert (result.content, result.error) == ("Plan updated", False)


async def test_ask_user_waits_for_the_users_answers():
    box = Toolbox("c")
    assert (await box.run("ask_user", {"questions": QUESTIONS})).waiting
    assert box.resume({"Which grade level?": "Grade 9"}).content == (
        'User has answered your questions: "Which grade level?"="Grade 9". '
        "You can now continue with the user's answers in mind.")
    assert box.resume(None).content == "User declined to answer questions"


async def test_ask_user_refuses_too_few_questions_or_options():
    box = Toolbox("c")
    for questions in ([], [{**QUESTIONS[0], "options": QUESTIONS[0]["options"][:1]}]):
        result = await box.run("ask_user", {"questions": questions})
        assert result.error and result.text().startswith("Invalid arguments for ask_user: questions"), result.content


async def test_unknown_tools_and_bad_arguments_come_back_as_errors():
    box = Toolbox("c")
    unknown = await box.run("teleport", {})
    assert (unknown.content, unknown.error) == ("Unknown tool: teleport", True)
    for arguments in ({"steps": PLAN}, {"plan": "draft it"}, {"plan": [{"step": "Draft", "status": "started"}]}):
        invalid = await box.run("todo", arguments)
        assert invalid.error and invalid.text().startswith("Invalid arguments for todo: plan"), invalid.content


def shape(schema: dict, defs: dict) -> dict:
    """The structure a schema gives a value, without its words: the type, enum, fields and items. A
    union counts as its first member that isn't null: X | None is an optional X, and edit's other
    members are the repairs pi accepts."""
    schema = defs[schema["$ref"].split("/")[-1]] if "$ref" in schema else schema
    if "anyOf" in schema:
        return shape(next(member for member in schema["anyOf"] if member.get("type") != "null"), defs)
    out = {key: schema[key] for key in ("type", "enum") if key in schema}
    if "properties" in schema:
        out["properties"] = {name: shape(field, defs) for name, field in schema["properties"].items()}
        out["required"] = sorted(schema.get("required", []))
    if "items" in schema:
        out["items"] = shape(schema["items"], defs)
    return out


async def test_each_schema_matches_its_function(llm):
    """The schema the model reads and the type hints its arguments are checked against are written
    apart, so this keeps them in step."""
    for name, (spec, function) in Toolbox("c", skills=SKILLS, llm=llm).tools.items():
        hinted = TypeAdapter(function).json_schema()
        written, expected = shape(spec["function"]["parameters"], {}), shape(hinted, hinted.get("$defs", {}))
        for field, hint in expected["properties"].items():
            if hint == {"type": "string"}:  # a closed list on a str the tool checks itself, like skill's names
                written["properties"].get(field, {}).pop("enum", None)
        assert written == expected, name


@pytest.mark.live
@pytest.mark.parametrize("candidate", [Model(provider="openai", model="gpt-5.6-luna"),
                                       Model(provider="anthropic", model="claude-haiku-5-5"),
                                       Model(provider="gemini", model="gemini-3.5-flash"),
                                       Model(provider="fireworks", model="accounts/fireworks/models/deepseek-v4p1-flash")],
                         ids=lambda m: m.provider)
async def test_every_provider_takes_the_tools(candidate, llm):
    specs = Toolbox("c", skills=SKILLS, llm=llm).specs
    candidate = candidate.model_copy(update={"config": Config(tools=specs, max_tokens=64)})
    await llm.query(candidate, [{"role": "user", "content": "Reply with the word ok."}])
