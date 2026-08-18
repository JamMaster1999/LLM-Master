"""web_search: ask for live search. See CONFIG.md.

Every provider's row in providers.py declares its search stance — adding a provider
without a search decision fails the registry's import-time checks.
"""
from typing import assert_never

from pydantic import BaseModel, Field

from llm_master import registry
from llm_master.config import UnsupportedField
from llm_master.providers import API, ProviderName


class Search(BaseModel, extra="forbid"):
    max_searches: int | None = Field(None, gt=0)


def translate(search: Search, api: API, provider: ProviderName) -> dict:
    stance = registry.search(provider)
    match stance:
        case "none":
            raise UnsupportedField(api, "web_search", f"{provider} has no search (its providers.py stance)")
        case "tool":
            pass
        case _:
            assert_never(stance)

    match api:
        case "completion":
            if provider not in ("gemini", "vertexai"):
                raise UnsupportedField(api, "web_search",
                                       f"no completion search spelling for {provider} — add it here")
            # no cap parameter exists; gemini grounding runs at most one search round, so any cap holds
            return {"tools": [{"google_search": {}}]}
        case "messages":
            tool: dict = {"type": "web_search_20260318", "name": "web_search"}
            if search.max_searches is not None:
                tool["max_uses"] = search.max_searches
            return {"tools": [tool]}
        case "responses":
            out: dict = {"tools": [{"type": "web_search"}]}
            if search.max_searches is not None:
                out["max_tool_calls"] = search.max_searches  # caps all built-ins; search is our only one
            return out
        case _:
            assert_never(api)
