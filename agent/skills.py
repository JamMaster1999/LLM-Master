"""The skill tool, Claude Code's Skill: its description lists every skill, and a call returns one
skill's instructions. A skill is an Agent Skills folder (agentskills.io): SKILL.md, whose
frontmatter gives the name and description, and any scripts, references and assets beside it."""
from pathlib import Path

import yaml

from agent.result import ToolError
from agent.sandbox import SKILLS

MAX_DESCRIPTION = 1024  # characters, the Agent Skills spec's limit

DESCRIPTION = ("Load a skill.\n\nA skill is a packaged set of instructions for a particular kind of task (building a "
               "Word document, writing a rubric), often with scripts and reference files. When the "
               "task at hand is one a listed skill covers, call this tool first: the skill's instructions load for you "
               f"to follow in place of your default approach. Its files are in the sandbox under {SKILLS}/<name>/, for "
               "bash and read. A skill already loaded in this conversation stays loaded; follow it rather than calling "
               "again.\n\nAvailable skills:\n")


class Skills:
    """A folder of skill folders, each loaded when the toolbox is built. The sandbox has its own copy
    at /skills."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        if link := next((path for path in self.folder.rglob("*") if path.is_symlink()), None):  # the sandbox mounts it all
            raise ValueError(f"Skills {self.folder}: {link} is a symlink; the sandbox gets real files only")
        self.skills = {path.name: _load(path) for path in sorted(self.folder.iterdir())
                       if path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"}
        listing = "\n".join(f"- {name}: {description}" for name, (description, _) in self.skills.items())
        self.spec = {"type": "function", "function": {
            "name": "skill",
            "description": DESCRIPTION + listing,
            "parameters": {"type": "object", "properties": {
                "skill": {"type": "string", "enum": list(self.skills), "description": "The name of a skill from the list"},
            }, "required": ["skill"]}}}

    def load(self, skill: str) -> str:
        """The skill's instructions, wrapped as agentskills.io suggests, with its other files listed
        but not read."""
        if skill not in self.skills:
            raise ToolError(f"Unknown skill: {skill}. Available skills: {', '.join(self.skills)}.")
        _, body = self.skills[skill]
        folder = self.folder / skill
        files = sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*")
                       if path.is_file() and path != folder / "SKILL.md")
        resources = "".join(f"<file>{file}</file>\n" for file in files)
        return (f'<skill_content name="{skill}">\n{body}\n\nSkill directory: {SKILLS}/{skill}\n'
                "Relative paths in this skill are relative to the skill directory.\n"
                + (f"<skill_resources>\n{resources}</skill_resources>\n" if files else "") + "</skill_content>")


def _load(folder: Path) -> tuple[str, str]:
    """A skill's description and instructions, from SKILL.md's YAML frontmatter and the text after it."""
    if not (folder / "SKILL.md").is_file():
        raise ValueError(f"Skill {folder}: SKILL.md is missing")
    lines = (folder / "SKILL.md").read_text().splitlines()
    if lines[:1] != ["---"] or "---" not in lines[1:]:
        raise ValueError(f"Skill {folder}: SKILL.md must start with frontmatter between --- lines")
    end = lines.index("---", 1)
    try:
        fields = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as error:
        raise ValueError(f"Skill {folder}: its frontmatter isn't valid YAML") from error
    if not isinstance(fields, dict):
        raise ValueError(f"Skill {folder}: its frontmatter must be a map of keys to values")
    if fields.get("name") != folder.name:
        raise ValueError(f"Skill {folder}: its name must be {folder.name}, the folder's name")
    description = fields.get("description")
    if not isinstance(description, str) or not 0 < len(description) <= MAX_DESCRIPTION:
        raise ValueError(f"Skill {folder}: its description is required, at most {MAX_DESCRIPTION} characters")
    return description, "\n".join(lines[end + 1:]).strip()
