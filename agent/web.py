"""The web tools on Exa: web_search is exa-py's own search tool, fetch_url its contents tool."""
import re
from collections.abc import Iterable
from functools import cached_property

from exa_py import AsyncExa

from agent.result import Result, ToolError
from agent.sandbox import Sandbox

EXA_API = "https://api.exa.ai"
PAGE_CHARACTERS = 10_000  # exa-py's default cut for page text
PAGES = "/workspace/.agent"  # where a longer page waits for read

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
                   "or mentioned by the user. Only those URLs, and links on pages you have read, can be fetched. "
                   "Pages are numbered like search results; cite them as [n].",
    "parameters": {"type": "object", "properties": {
        "urls": {"type": "array", "items": {"type": "string"}, "description": "Absolute URLs of the pages to read, "
                 "including the scheme. Pass several URLs to read them in a single call."},
    }, "required": ["urls"]}}}

_URL = re.compile(r"https?://[^\s<>\"'()\[\]]+")


class Web:
    """Search and fetch for one conversation. Sources are numbered in the order they first appear,
    and fetch_url reads only URLs the conversation has already seen, so model-written code can't
    smuggle sandbox data out in a URL of its own making."""

    def __init__(self, sandbox: Sandbox, sources: list[dict], urls: Iterable[str]) -> None:
        self.sandbox = sandbox
        self.sources = sources
        self.seen = {_key(url) for url in urls} | {_key(part["url"]) for part in sources if "url" in part}

    @cached_property
    def exa(self) -> AsyncExa:  # made on first use, so a conversation that never searches needs no key
        return AsyncExa(api_base=EXA_API)

    async def search(self, query: str, objective: str, type: str = "auto") -> Result:
        if type not in ("auto", "deep"):
            raise ToolError("type must be auto or deep.")
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
        unseen = [url for url in urls if _key(url) not in self.seen]
        if unseen:
            raise ToolError(f"fetch_url reads only URLs the user gave, earlier results returned, or pages you read "
                            f"link to. Not seen: {', '.join(unseen)}. Use web_search to find pages.")
        data = await self._post("/contents", {"urls": urls, "text": True})
        pages = {page["id"]: page for page in data["results"]}
        errors = {status["id"]: status.get("error") or {} for status in data.get("statuses") or []}
        entries, parts = [], []
        for url in urls:
            if url not in pages:
                entries.append(f"Error fetching {url}: {errors.get(url, {}).get('tag', 'no content')}")
                continue
            page, part = pages[url], self._number(pages[url])
            parts.append(part)
            text = page.get("text") or ""
            self.seen |= {_key(link) for link in _URL.findall(text)}
            if len(text) > PAGE_CHARACTERS:
                saved = f"{PAGES}/page-{part['sourceId']}.txt"
                await self.sandbox.write(saved, text.encode())
                text = (text[:PAGE_CHARACTERS] + f"\n\n[Cut at {PAGE_CHARACTERS} of {len(text)} characters. The whole "
                        f"page is at {saved.removeprefix('/workspace/')}; read it with offset and limit.]")
            entries.append(_entry(part, page) + "\n" + text)
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
        self.seen.add(_key(result["url"]))
        for part in self.sources:
            if part.get("url") == result["url"]:
                return part
        part = {"type": "source-url", "sourceId": str(len(self.sources) + 1), "url": result["url"],
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


def _key(url: str) -> str:
    """A URL as fetch_url compares it: no fragment, no trailing slash."""
    return url.split("#")[0].rstrip("/")
