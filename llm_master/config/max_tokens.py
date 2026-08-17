"""max_tokens: one rename on responses. Card 12 in CONFIG.md."""
from llm_master.config import API


def translate(n: int, api: API) -> dict:
    return {"max_output_tokens": n} if api == "responses" else {"max_tokens": n}
