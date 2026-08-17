"""timeout: seconds, forwarded to each provider SDK. Card 13 in CONFIG.md."""
from llm_master.config import API


def translate(seconds: float, api: API) -> dict:
    return {"timeout": seconds}
