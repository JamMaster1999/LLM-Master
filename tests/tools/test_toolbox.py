"""The toolbox's own behaviour, for the tools that need neither Modal nor Exa."""
import inspect

import pytest

from agent import Toolbox, files, loop, shell, web

PLAN = [{"step": "Draft the quiz", "status": "in_progress"}, {"step": "Write the key", "status": "pending"}]
QUESTIONS = [{"question": "Which grade level?", "header": "Grade", "multiSelect": False,
              "options": [{"label": "Grade 9", "description": "Ninth grade"},
                          {"label": "Grade 10", "description": "Tenth grade"}]}]


async def test_todo_acknowledges_the_plan():
    result = await Toolbox("c").run("todo", {"plan": PLAN})
    assert (result.content, result.error) == ("Plan updated", False)


async def test_ask_user_waits_for_the_users_answers():
    box = Toolbox("c")
    assert (await box.run("ask_user", {"questions": QUESTIONS})).waiting == "question"
    assert (await box.resume("ask_user", {"questions": QUESTIONS}, {"Which grade level?": "Grade 9"})).content == (
        'User has answered your questions: "Which grade level?"="Grade 9". '
        "You can now continue with the user's answers in mind.")
    assert (await box.resume("ask_user", {"questions": QUESTIONS}, None)).content == "User declined to answer questions"


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


BOUND = {"self", "sandbox", "run", "tools"}  # what the toolbox binds before the model's arguments


@pytest.mark.parametrize("spec, function", [
    (files.READ, files.read), (files.WRITE, files.write), (files.EDIT, files.edit), (shell.BASH, shell.bash),
    (shell.RUN_PROGRAM, shell.run_program), (web.SEARCH, web.Web.search), (web.FETCH, web.Web.fetch),
    (files.DELIVER, files.deliver_file), (loop.TODO, loop.todo), (loop.ASK_USER, loop.ask_user)])
def test_each_schema_matches_its_function(spec, function):
    """The schema the model reads and the signature its arguments are checked against are written
    apart, so this keeps them in step."""
    schema, parameters = spec["function"]["parameters"], inspect.signature(function).parameters
    assert set(schema["properties"]) == set(parameters) - BOUND
    assert set(schema["required"]) == {name for name, p in parameters.items() if p.default is p.empty} - BOUND
