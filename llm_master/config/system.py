"""system: the system prompt. Card 1 in CONFIG.md."""
from llm_master.config import API


def translate(text: str, api: API) -> dict:
    match api:
        case "completion":
            return {}  # no param exists: prepended as a system-role message at assembly (io/messages)
        case "messages":
            return {"system": text}
        case "responses":
            return {"instructions": text}
