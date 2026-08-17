"""temperature: same name on every API. Card 5 in CONFIG.md."""
from llm_master.config import API


def translate(t: float, api: API) -> dict:
    return {"temperature": t}
