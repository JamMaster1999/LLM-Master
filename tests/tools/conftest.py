"""A local server standing in for Exa, for the web tools' rules and for long replies."""
import json
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import pytest

from agent import web


@pytest.fixture
def exa(monkeypatch, request) -> dict[str, str]:
    """Answers /search with pages B and C, and /contents with whatever pages were asked for. A page's
    text comes from the dict this returns, "page text" by default."""
    texts: dict[str, str] = {}

    class Answer(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/search":
                results = [{"id": url, "url": url, "title": url[8], "highlights": [f"about {url[8]}"]}
                           for url in ("https://b.example", "https://c.example")]
            else:
                results = [{"id": url, "url": url, "title": "Page", "text": texts.get(url, "page text")}
                           for url in body["urls"]]
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
    return texts
