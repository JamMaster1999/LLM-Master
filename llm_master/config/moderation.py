"""moderation: free openai pre-check before dispatch. See CONFIG.md."""
from any_llm import AnyLLM


class Flagged(Exception):
    """Input failed moderation; nothing was dispatched or billed."""

    def __init__(self, categories: list[str]):
        self.categories = categories
        super().__init__(f"input flagged: {', '.join(categories)}")


async def check(openai: AnyLLM, text: str) -> None:
    result = (await openai.amoderation("omni-moderation-latest", text)).results[0]
    if result.flagged:
        raise Flagged(sorted(k for k, v in result.categories.items() if v))
