"""moderation: free openai pre-check before dispatch. See CONFIG.md."""
from any_llm import AnyLLM

from llm_master.io.messages import Message
from llm_master.io.parts import Text


class Flagged(Exception):
    """Input failed moderation; nothing was dispatched or billed."""

    def __init__(self, categories: list[str]):
        self.categories = categories
        super().__init__(f"input flagged: {', '.join(categories)}")


async def check(openai: AnyLLM, history: list[Message]) -> None:
    """Reads the newest user message's text, as v1 did."""
    content = next((m.content for m in reversed(history) if m.role == "user"), "")
    text = content if isinstance(content, str) else "\n".join(p.text for p in content if isinstance(p, Text))
    result = (await openai.amoderation("omni-moderation-latest", text)).results[0]
    if result.flagged:
        raise Flagged(sorted(k for k, v in result.categories.items() if v))
