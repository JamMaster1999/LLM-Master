"""What a tool hands back: content for the model, and files and sources for the user's UI."""
from dataclasses import dataclass, field


def whole(text: str) -> str:
    """The text with its split UTF-16 halves rejoined and a lone one replaced: JSON encoders and
    providers refuse lone halves, and some tools' sources hand them over."""
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


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
    waiting: bool = False  # no result yet: the user answers, Toolbox.resume finishes it

    def text(self) -> str:
        """The content's text, for callers that take text only."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part["text"] for part in self.content if part["type"] == "text")
