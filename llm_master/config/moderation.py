"""moderation: free openai pre-check before dispatch. See CONFIG.md."""
from typing import TYPE_CHECKING

from llm_master.io.messages import Message
from llm_master.io.parts import Text

if TYPE_CHECKING:
    from llm_master.facade import LLM

MODEL = "omni-moderation-latest"


class Flagged(Exception):
    """Input failed moderation; nothing was dispatched or billed."""

    def __init__(self, categories: list[str]):
        self.categories = categories
        super().__init__(f"input flagged: {', '.join(categories)}")


async def check(llm: "LLM", history: list[Message]) -> None:
    """Reads the newest user message's text, as v1 did. It waits in the model's own limiter window."""
    content = next((m.content for m in reversed(history) if m.role == "user"), "")
    text = content if isinstance(content, str) else "\n".join(p.text for p in content if isinstance(p, Text))
    await llm.limiter.wait("openai", MODEL)
    result = (await llm.client("openai").amoderation(MODEL, text)).results[0]
    if result.flagged:
        raise Flagged(sorted(k for k, v in result.categories.items() if v))
