"""The skill tool: the listing in its description, a skill's instructions, and its files in the sandbox."""
import uuid
from pathlib import Path

import pytest

from agent import Toolbox
from agent.sandbox import delete

SKILLS = Path(__file__).parent / "skills"


def test_skills_are_listed_in_the_tools_description(tmp_path):
    spec = Toolbox("c", skills=SKILLS).specs[-1]["function"]
    assert spec["description"].endswith(
        "Available skills:\n- greet: Greets a student by name: use when a lesson starts.\n- quiz: Writes a one-question quiz.")
    assert spec["parameters"]["properties"]["skill"]["enum"] == ["greet", "quiz"]
    for box in (Toolbox("c"), Toolbox("c", skills=tmp_path)):  # no skills, or an empty folder: no tool
        assert "skill" not in [spec["function"]["name"] for spec in box.specs]


async def test_a_skill_comes_back_with_its_directory_and_files():
    box = Toolbox("c", skills=SKILLS)
    assert (await box.run("skill", {"skill": "greet"})).content == (
        '<skill_content name="greet">\n# Greet\n\nRun `scripts/greet.py <name>` and say what it prints.\n\n'
        "Skill directory: /skills/greet\nRelative paths in this skill are relative to the skill directory.\n"
        "<skill_resources>\n<file>scripts/greet.py</file>\n</skill_resources>\n</skill_content>")
    assert (await box.run("skill", {"skill": "quiz"})).content == (
        '<skill_content name="quiz">\nAsk exactly one question.\n\nSkill directory: /skills/quiz\n'
        "Relative paths in this skill are relative to the skill directory.\n</skill_content>")


async def test_an_unknown_skill_is_an_error_result():
    result = await Toolbox("c", skills=SKILLS).run("skill", {"skill": "essay"})
    assert (result.content, result.error) == ("Unknown skill: essay. Available skills: greet, quiz.", True)


@pytest.mark.parametrize("skill_md", [
    "# Broken\n",
    "---\nname: other\ndescription: Named for another folder.\n---\n",
    f"---\nname: broken\ndescription: {'x' * 1025}\n---\n",
])
def test_a_broken_skill_fails_construction(tmp_path, skill_md):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text(skill_md)
    with pytest.raises(ValueError, match="^Skill .*/broken: "):
        Toolbox("c", skills=tmp_path)


@pytest.fixture
async def box():
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}", skills=SKILLS)
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


@pytest.mark.live
async def test_bash_runs_a_skills_script_in_the_sandbox(box):
    result = await box.run("bash", {"command": "python /skills/greet/scripts/greet.py Ada"})
    assert result.content == "Hello, Ada!"
