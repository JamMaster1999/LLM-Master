"""The caller's own tools: listed, run and failed exactly like the built-in ones."""
import uuid

import pytest

from agent import Toolbox, ToolError
from agent.sandbox import delete


def lookup_grade(student: str) -> str:
    """The student's current grade in this course."""
    if student == "nobody":
        raise ToolError("No student named nobody.")
    if student == "crash":
        raise RuntimeError("database password is hunter2")
    return f"{student}: A-"


async def count_submissions(assignment: str) -> int:
    """How many students submitted the assignment."""
    return 24


async def test_custom_tools_are_listed_and_run_like_built_ins():
    box = Toolbox("c", tools=[lookup_grade, count_submissions])
    assert [spec["function"]["name"] for spec in box.specs][-2:] == ["lookup_grade", "count_submissions"]
    assert (await box.run("lookup_grade", {"student": "Ada"})).content == "Ada: A-"
    assert (await box.run("count_submissions", {"assignment": "Quiz 3"})).content == "24"
    assert (await box.run("lookup_grade", {"name": "Ada"})).text().startswith("Invalid arguments for lookup_grade:")


async def test_custom_tool_failures_read_like_built_in_ones():
    box = Toolbox("c", tools=[lookup_grade])
    refused = await box.run("lookup_grade", {"student": "nobody"})
    assert (refused.content, refused.error) == ("No student named nobody.", True)
    crashed = await box.run("lookup_grade", {"student": "crash"})
    assert (crashed.content, crashed.error) == ("lookup_grade failed (RuntimeError). The details are in the server log.", True)


async def test_a_custom_tool_cant_take_a_built_in_name():
    def read(path: str) -> str:
        """Shadows the built-in."""
        return ""
    with pytest.raises(ValueError, match="share its name"):
        Toolbox("c", tools=[read])


@pytest.fixture
async def box():
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}", tools=[lookup_grade])
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


@pytest.mark.live
async def test_programs_call_custom_tools_up_to_the_cap(box):
    code = """
grades = await asyncio.gather(*[lookup_grade({"student": f"s{i}"}) for i in range(100)])
print(len(grades), grades[0])
try:
    await lookup_grade(student="one too many")
except ToolError as error:
    print(error)
"""
    assert (await box.run("run_program", {"code": code})).content == "100 s0: A-\nrun_program allows at most 100 tool calls."
