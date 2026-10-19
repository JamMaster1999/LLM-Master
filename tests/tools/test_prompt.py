"""The system prompt, which the backend sends with every request."""
import re
from pathlib import Path

from agent import Toolbox

PROMPT = Path(__file__).parents[2] / "prompts" / "system.md"


def test_the_system_prompt_names_only_tools_the_toolbox_has(llm):
    tools = {spec["function"]["name"] for spec in Toolbox("c", skills=Path(__file__).parent / "skills", llm=llm).specs}
    prompt = PROMPT.read_text()
    named = set(re.findall(r"`([a-z_]+)`", prompt)) | set(re.findall(r"\b[a-z]+_[a-z_]+\b", prompt))
    assert named and named <= tools, named - tools
