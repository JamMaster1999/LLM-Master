"""The web tools on Exa: web_search is exa-py's own search tool, fetch_url its contents tool."""
from collections.abc import Iterable
from functools import cached_property
from typing import Literal

from exa_py import AsyncExa

from agent.result import Result, ToolError

EXA_API = "https://api.exa.ai"

SEARCH = {"type": "function", "function": {
    "name": "web_search",
    "description": "Search the web for up-to-date, relevant information. Describe the ideal page rather than listing "
                   "keywords. Results are numbered; cite the ones you use as [n].",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Natural language search query. Should be a semantically rich "
                  "description of the ideal page, not just keywords."},
        "objective": {"type": "string", "description": "Goal for this search turn; say which documents should rank "
                      "first, which should be excluded, and what specific facts or figures to pull from them."},
        "type": {"type": "string", "enum": ["auto", "deep"], "description": "auto (default) answers in about a "
                 "second. deep runs a multi-step search, 4-15 seconds, for questions one search can't cover."},
    }, "required": ["query", "objective"]}}}

FETCH = {"type": "function", "function": {
    "name": "fetch_url",
    "description": "Read the full contents of web pages you already have URLs for, such as pages returned by a search "
                   "or mentioned by the user. Only those URLs can be fetched; search for anything else. Pages are "
                   "numbered like search results; cite them as [n].",
    "parameters": {"type": "object", "properties": {
        "urls": {"type": "array", "items": {"type": "string"}, "description": "Absolute URLs of the pages to read, "
                 "including the scheme. Pass several URLs to read them in a single call."},
    }, "required": ["urls"]}}}

class Web:
    """Search and fetch for one conversation. Sources are numbered in the order they first appear.
    fetch_url reads only URLs the user wrote or Exa returned, never one the model made up, so it
    can't carry the conversation's data out."""

    def __init__(self, sources: list[dict], urls: Iterable[str]) -> None:
        self.sources = sources
        self.seen = set(urls) | {part["url"] for part in sources if "url" in part}

    @cached_property
    def exa(self) -> AsyncExa:  # made on first use, so a conversation that never searches needs no key
        return AsyncExa(api_base=EXA_API)

    async def search(self, query: str, objective: str, type: Literal["auto", "deep"] = "auto") -> Result:
        data = await self._post("/search", {"query": query, "objective": objective, "type": type,
                                            "numResults": 10, "contents": {"highlights": True}})
        if not data["results"]:
            return Result("No search results found.")
        entries, parts = [], []
        for result in data["results"]:
            part = self._number(result)
            parts.append(part)
            entries.append(_entry(part, result) + "Highlights:\n" + "\n".join(result.get("highlights") or []))
        text = "\n\n---\n\n".join(entries)
        for notice in data.get("notices") or []:
            text += f"\n\nNotice: {notice['message']}"
        return Result(text, sources=parts)

    async def fetch(self, urls: list[str]) -> Result:
        unseen = [url for url in urls if url not in self.seen]  # exact: a changed slash could carry data
        if unseen:
            raise ToolError(f"fetch_url reads only URLs the user gave or earlier results returned. "
                            f"Not seen: {', '.join(unseen)}. Use web_search to find pages.")
        data = await self._post("/contents", {"urls": urls, "text": True})
        pages = {page["id"]: page for page in data["results"]}
        errors = {status["id"]: status.get("error") or {} for status in data.get("statuses") or []}
        entries, parts = [], []
        for url in urls:
            if url not in pages:
                entries.append(f"Error fetching {url}: {errors.get(url, {}).get('tag', 'no content')}")
                continue
            part = self._number(pages[url])
            parts.append(part)
            entries.append(_entry(part, pages[url]) + "\n" + (pages[url].get("text") or ""))
        return Result("\n\n---\n\n".join(entries), sources=parts, error=not parts)

    async def aclose(self) -> None:
        if "exa" in self.__dict__:
            await self.exa.client.aclose()

    async def _post(self, endpoint: str, body: dict) -> dict:
        """Exa's raw JSON: exa-py's parsed responses drop the notices Exa asks us to pass on."""
        data = await self.exa.async_request(endpoint, body)
        assert isinstance(data, dict)  # a dict unless the body asks for a stream
        return data

    def _number(self, result: dict) -> dict:
        """The source part for a result: the one already numbered for its URL, or a new one."""
        self.seen.add(result["url"])
        for part in self.sources:
            if part.get("url") == result["url"]:
                return part
        last = max((int(part["sourceId"]) for part in self.sources), default=0)  # numbers may have gaps
        part = {"type": "source-url", "sourceId": str(last + 1), "url": result["url"],
                "title": result.get("title") or result["url"]}
        self.sources.append(part)
        return part


def _entry(part: dict, result: dict) -> str:
    lines = [f"[{part['sourceId']}] Title: {result.get('title') or ''}", f"URL: {result['url']}"]
    if result.get("publishedDate"):
        lines.append(f"Published: {result['publishedDate'][:10]}")
    if result.get("author"):
        lines.append(f"Author: {result['author']}")
    return "\n".join(lines) + "\n"
