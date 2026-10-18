"""The toolbox's own behaviour, for the tools that need neither Modal nor Exa."""
from agent import Toolbox

PLAN = [{"step": "Draft the quiz", "status": "in_progress"}, {"step": "Write the key", "status": "pending"}]
QUESTIONS = [{"question": "Which grade level?", "header": "Grade", "multiSelect": False,
              "options": [{"label": "Grade 9", "description": "Ninth grade"},
                          {"label": "Grade 10", "description": "Tenth grade"}]}]


async def test_specs_list_every_built_in_tool():
    assert [spec["function"]["name"] for spec in Toolbox("c").specs] == [
        "read", "write", "edit", "bash", "run_program", "web_search", "fetch_url", "deliver_file", "todo", "ask_user"]


async def test_todo_keeps_no_state():
    result = await Toolbox("c").run("todo", {"plan": PLAN})
    assert (result.content, result.error) == ("Plan updated", False)


async def test_ask_user_waits_for_the_users_answers():
    box = Toolbox("c")
    assert (await box.run("ask_user", {"questions": QUESTIONS})).waiting == "question"
    assert (await box.resume("ask_user", {"questions": QUESTIONS}, {"Which grade level?": "Grade 9"})).content == (
        'User has answered your questions: "Which grade level?"="Grade 9". '
        "You can now continue with the user's answers in mind.")
    assert (await box.resume("ask_user", {"questions": QUESTIONS}, None)).content == "User declined to answer questions"


async def test_unknown_tools_and_bad_arguments_come_back_as_errors():
    box = Toolbox("c")
    unknown = await box.run("teleport", {})
    assert (unknown.content, unknown.error) == ("Unknown tool: teleport", True)
    invalid = await box.run("todo", {"steps": PLAN})
    assert invalid.error and invalid.text().startswith("Invalid arguments for todo:")
