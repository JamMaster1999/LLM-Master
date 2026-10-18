"""Parts read their source once, at creation; the bytes decide what an image is."""
import base64
import functools
import io
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer

import pytest
from PIL import Image as Pillow

from llm_master.io.parts import Audio, File, Image, PartError


def png() -> bytes:
    out = io.BytesIO()
    Pillow.new("RGB", (4, 4), "red").save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def served(tmp_path, request):
    """tmp_path over HTTP: its files at their names, anything else a 404."""
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    request.addfinalizer(server.server_close)
    request.addfinalizer(server.shutdown)
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_path_url_and_data_uri_give_the_same_part(tmp_path, served):
    (tmp_path / "a.png").write_bytes(png())
    from_uri = Image("data:image/png;base64," + base64.b64encode(png()).decode())
    assert Image(str(tmp_path / "a.png")) == Image(f"{served}/a.png") == from_uri
    assert base64.b64decode(from_uri.data) == png() and from_uri.mime == "image/png"


def test_the_bytes_decide_the_image_type(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(png())
    assert Image(str(tmp_path / "photo.jpg")).mime == "image/png"


def test_heic_becomes_jpeg(tmp_path):
    path = tmp_path / "photo.heic"
    Pillow.new("RGB", (4, 4), "blue").save(path)  # pillow-heif registered the HEIC writer
    part = Image(str(path))
    assert part.mime == "image/jpeg"
    assert Pillow.open(io.BytesIO(base64.b64decode(part.data))).format == "JPEG"


def test_file_and_audio_take_their_names_from_the_source(tmp_path):
    (tmp_path / "report#1.pdf").write_bytes(b"%PDF-1.4 x")
    (tmp_path / "clip.flac").write_bytes(b"fLaC")
    assert File(str(tmp_path / "report#1.pdf")).filename == "report#1.pdf"
    assert Audio(str(tmp_path / "clip.flac")).format == "flac"
    assert Audio("data:audio/mpeg;base64,SUQz").format == "mp3"  # no name: the declared type's short name


@pytest.mark.parametrize("make, source", [
    (Image, "{tmp}/missing.png"),
    (Image, "{url}/gone.png"),
    (Image, "{tmp}/notes.png"),
    (Image, "data:image/png;base64,é"),
    (File, "{tmp}/notes.png"),
], ids=["no-file", "http-404", "not-an-image", "bad-base64", "not-a-pdf"])
def test_unreadable_source_raises_at_creation(make, source, tmp_path, served):
    (tmp_path / "notes.png").write_text("not media")
    with pytest.raises(PartError):
        make(source.format(tmp=tmp_path, url=served))
