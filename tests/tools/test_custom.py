"""The caller's own tools: listed, run and failed exactly like the built-in ones."""
import uuid

import pytest
from pydantic import TypeAdapter

from agent import Toolbox, ToolError
from agent.sandbox import delete


def lookup_grade(student: str) -> str:
    """The student's current grade in this course."""
    if student == "nobody":
        raise ToolError("No student named nobody.")
    if student == "crash":  # a bug in the tool: its own pydantic check fails, and the message holds a secret
        TypeAdapter(int).validate_python("database password is hunter2")
    return f"{student}: A-"


async def count_submissions(assignment: str) -> dict:
    """How many students submitted the assignment, on time and late."""
    return {"on_time": 21, "late": 3}


async def test_custom_tools_are_listed_and_run_like_built_ins():
    box = Toolbox("c", tools=[lookup_grade, count_submissions])
    assert [spec["function"]["name"] for spec in box.specs][-2:] == ["lookup_grade", "count_submissions"]
    assert (await box.run("lookup_grade", {"student": "Ada"})).content == "Ada: A-"
    assert (await box.run("count_submissions", {"assignment": "Quiz 3"})).content == '{"on_time": 21, "late": 3}'
    assert (await box.run("lookup_grade", {"name": "Ada"})).text().startswith("Invalid arguments for lookup_grade:")


async def test_custom_tool_failures_read_like_built_in_ones(caplog):
    box = Toolbox("c", tools=[lookup_grade])
    refused = await box.run("lookup_grade", {"student": "nobody"})
    assert (refused.content, refused.error) == ("No student named nobody.", True)
    crashed = await box.run("lookup_grade", {"student": "crash"})
    assert (crashed.content, crashed.error) == (
        "lookup_grade failed (ValidationError). The details are in the server log.", True)
    assert "hunter2" in caplog.text


async def test_a_custom_tool_cant_take_a_built_in_name():
    def read(path: str) -> str:
        """Shadows the built-in."""
        return ""
    with pytest.raises(ValueError, match="repeated: read$"):
        Toolbox("c", tools=[read])


@pytest.fixture
async def box():
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}", tools=[lookup_grade])
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


@pytest.mark.live
async def test_programs_call_custom_tools(box):
    code = 'print(await lookup_grade({"student": "Ada"}))'
    assert (await box.run("run_program", {"code": code})).content == "Ada: A-"
