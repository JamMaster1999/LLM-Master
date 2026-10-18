"""Parts read their source once, at creation: same bytes from a path, a URL or a data-uri."""
import base64
import io
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import pytest
from PIL import Image as Pillow

from llm_master.io.parts import Audio, File, Image, PartError


def png() -> bytes:
    out = io.BytesIO()
    Pillow.new("RGB", (4, 4), "red").save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def url(request):
    """Serves PNG bytes at /a.png; any other path is a 404."""
    class Serve(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/a.png":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(png())

        def log_message(self, format, *args):
            pass

    server = ThreadingTCPServer(("127.0.0.1", 0), Serve)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    request.addfinalizer(server.server_close)
    request.addfinalizer(server.shutdown)
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_path_url_and_data_uri_give_the_same_part(tmp_path, url):
    path = tmp_path / "a.png"
    path.write_bytes(png())
    from_uri = Image("data:image/png;base64," + base64.b64encode(png()).decode())
    assert Image(str(path)) == Image(f"{url}/a.png") == from_uri
    assert base64.b64decode(from_uri.data) == png() and from_uri.mime == "image/png"


def test_heic_becomes_png(tmp_path):
    path = tmp_path / "photo.heic"
    Pillow.new("RGB", (4, 4), "blue").save(path)  # pillow-heif registered the HEIC writer
    part = Image(str(path))
    assert part.mime == "image/png"
    assert Pillow.open(io.BytesIO(base64.b64decode(part.data))).format == "PNG"


def test_file_and_audio_take_their_names_from_the_source(tmp_path):
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 x")
    (tmp_path / "clip.mp3").write_bytes(b"ID3")
    assert File(str(tmp_path / "report.pdf")).filename == "report.pdf"
    assert Audio(str(tmp_path / "clip.mp3")).format == "mp3"


@pytest.mark.parametrize("make, source", [
    (Image, "{tmp}/missing.png"),
    (Image, "{url}/gone.png"),
    (Image, "{tmp}/notes.txt"),
    (File, "{tmp}/notes.txt"),
])
def test_unreadable_source_raises_at_creation(make, source, tmp_path, url):
    (tmp_path / "notes.txt").write_text("not media")
    with pytest.raises(PartError):
        make(source.format(tmp=tmp_path, url=url))
