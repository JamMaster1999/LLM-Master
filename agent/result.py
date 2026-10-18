"""What a tool hands back: content for the model, and files and sources for the user's UI."""
from dataclasses import dataclass, field
from typing import Literal


class ToolError(Exception):
    """A failure the model caused and can fix; the model reads the message."""


@dataclass
class UserFile:
    """A file for the user. The caller hosts the bytes and sends an AI SDK `file` part."""

    name: str
    mime: str
    data: bytes


@dataclass
class Result:
    content: str | list[dict]  # the tool message's content: text, or IO.md's text and image parts
    files: list[UserFile] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # AI SDK source parts; sourceId is the number the model cites
    error: bool = False
    waiting: Literal["question", "approval"] | None = None  # no result yet: the user replies, Toolbox.resume finishes it

    def text(self) -> str:
        """The content's text, for callers that take text only."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part["text"] for part in self.content if part["type"] == "text")
