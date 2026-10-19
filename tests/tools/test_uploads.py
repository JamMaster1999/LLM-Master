"""Uploads and views through box.attach and read, on real Modal. Together is live, or a local stand-in."""
import asyncio
import base64
import email
import email.policy
import json
import os
import pathlib
import threading
import uuid
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import pytest

from agent import Toolbox, transcribe
from agent.sandbox import delete

pytestmark = pytest.mark.live

TWO_PAGES = '<p>First page</p><p style="page-break-before: always">Second page</p>'
DOCUMENT = (f"printf '%s' '{TWO_PAGES}' > pages.html && "
            "soffice --headless --infilter='HTML (StarWriter)' --convert-to docx pages.html >/dev/null && "
            "soffice --headless --convert-to pdf pages.docx >/dev/null")
NOTEBOOK = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Quiz\n", "Name the gas plants give off."]},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": ["answer = 'oxygen'"]}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


@pytest.fixture
async def box():
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}")
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


async def bash(box: Toolbox, command: str) -> str:
    result = await box.run("bash", {"command": command})
    assert not result.error, result.content
    return result.text()


async def made(box: Toolbox, command: str, path: str) -> bytes:
    """A file made in the sandbox, as the bytes of an upload."""
    await bash(box, command)
    return await box.sandbox.read(path)


def texts(parts: list[dict]) -> list[str]:
    return [part["text"] if part["type"] == "text" else part["mime"] for part in parts]


async def test_a_text_upload_shows_its_text_and_a_taken_name_gets_a_number(box):
    assert texts(await box.attach("notes.txt", b"one\ntwo\n")) == ["Attached file: uploads/notes.txt\n\none\ntwo"]
    [long] = texts(await box.attach("notes.txt", "\n".join(map(str, range(1, 3001))).encode()))
    assert long.startswith("Attached file: uploads/notes (2).txt\n\n1\n2\n")
    assert long.endswith("[Showing lines 1-2000 of 3000. Use offset=2001 to continue, "
                         "or search uploads/notes (2).txt with bash (rg, grep).]")


async def test_text_decodes_from_utf16_and_legacy_encodings_and_binary_is_not_supported(box):
    french = "Élève,Remarque\nAnaïs,très bien réussi, même si elle doit revoir les étapes du cycle\n"
    assert texts(await box.attach("names.txt", "Zoë\n".encode("utf-16"))) == ["Attached file: uploads/names.txt\n\nZoë"]
    assert texts(await box.attach("marks.csv", french.encode("cp1252"))) == [
        f"Attached file: uploads/marks.csv\n\n{french.strip()}"]
    assert texts(await box.attach("blob.bin", b"PK\x03\x04\xff\xfe")) == [
        "Attached file: uploads/blob.bin\n\nBinary files aren't supported yet."]


async def test_images_spreadsheets_and_video_get_no_view(box):
    avif = await made(box, "python -c \"from PIL import Image; Image.new('RGB', (40, 40)).save('a.avif')\"", "a.avif")
    assert texts(await box.attach("photo.avif", avif)) == ["Attached file: uploads/photo.avif", "image/jpeg"]
    assert texts(await box.attach("grades.xlsx", b"PK\x03\x04")) == ["Attached file: uploads/grades.xlsx"]
    assert texts(await box.attach("lesson.mp4", b"\x00\x00\x00\x18ftypmp42")) == [
        "Attached file: uploads/lesson.mp4\n\nVideo files aren't supported yet."]
    assert await bash(box, "test -e .agent/views || echo none") == "none"


async def test_a_document_gets_a_view_that_attach_shows_and_read_pages_through_by_lines(box):
    docx = await made(box, DOCUMENT, "pages.docx")
    assert texts(await box.attach("pages.docx", docx)) == [
        "Attached file: uploads/pages.docx (2 pages)\n\n## Page 1\nFirst page\n\n## Page 2\nSecond page"]
    view = await bash(box, "ls -d .agent/views/*")
    assert await bash(box, f"cd {view} && find . -type f | sort") == (
        "./pages/1.jpg\n./pages/2.jpg\n./source.pdf\n./transcript.jsonl")
    assert (await box.run("read", {"path": "uploads/pages.docx", "offset": 4, "limit": 2})).content == (
        "## Page 2\nSecond page")


async def test_a_long_transcript_is_searched_rather_than_shown(box):
    chapters = "<p style='page-break-before: always'></p>".join(f"<p>{'Chlorophyll absorbs red and blue light. ' * 60}</p>"
                                                               for _ in range(25))
    docx = await made(box, f"printf '%s' \"{chapters}\" > long.html && soffice --headless "
                           "--infilter='HTML (StarWriter)' --convert-to docx long.html >/dev/null", "long.docx")
    [text] = texts(await box.attach("long.docx", docx))
    view = await bash(box, "ls -d .agent/views/*")
    assert text == (f"Attached file: uploads/long.docx (25 pages)\n\nIts transcript is too long to show here. Search "
                    f"{view}/transcript.jsonl with bash (rg, grep), and read uploads/long.docx with offset and limit, "
                    f"or with pages to see page images.")
    cut = (await box.run("read", {"path": "uploads/long.docx"})).text()
    assert cut.startswith("## Page 1\nChlorophyll") and cut.endswith(f"search {view}/transcript.jsonl with bash (rg, grep).]")


async def test_read_shows_pages_as_images(box):
    await bash(box, DOCUMENT + " && echo hi > notes.txt")
    label, image = (await box.run("read", {"path": "pages.pdf", "pages": "2"})).content
    assert label == {"type": "text", "text": "Page 2 of 2"}
    assert base64.b64decode(image["data"])[:2] == b"\xff\xd8"
    for path, pages, message in [("pages.pdf", "1-11", 'pages "1-11" asks for 11 pages. Ask for at most 10 per call.'),
                                 ("pages.pdf", "3", "pages.pdf has 2 pages."),
                                 ("pages.pdf", "two", 'Invalid pages "two".'),
                                 ("notes.txt", "1", "notes.txt has no pages.")]:
        result = await box.run("read", {"path": path, "pages": pages})
        assert result.error and result.text().startswith(message), result.content


async def test_a_scan_is_read_with_ocr(box):
    scan = await made(box, DOCUMENT + " && pdftoppm -r 150 -gray pages.pdf page && python -c \"import glob; "
                      "from PIL import Image; pages = [Image.open(f) for f in sorted(glob.glob('page*.pgm'))]; "
                      "pages[0].save('scan.pdf', save_all=True, append_images=pages[1:])\"", "scan.pdf")
    [text] = texts(await box.attach("scan.pdf", scan))
    assert text == "Attached file: uploads/scan.pdf (2 pages)\n\n## Page 1\nFirst page\n\n## Page 2\nSecond page"


async def test_views_stop_at_100_pages(box):
    await bash(box, "python -c \"from PIL import Image; page = Image.new('L', (200, 300), 255); "
                    "[page.save(f'{n}.pdf', save_all=True, append_images=[page] * (n - 1)) for n in (100, 101)]\"")
    [hundred] = texts(await box.attach("hundred.pdf", await box.sandbox.read("100.pdf")))
    assert hundred.startswith("Attached file: uploads/hundred.pdf (100 pages)\n\n## Page 1\n")
    assert texts(await box.attach("long.pdf", await box.sandbox.read("101.pdf"))) == [
        "Attached file: uploads/long.pdf\n\nuploads/long.pdf has 101 pages, too long for read, which takes 100 at "
        "most. Use bash to work with it."]


async def test_parallel_reads_of_a_new_file_build_one_view(box):
    await bash(box, DOCUMENT)
    results = await asyncio.gather(*[box.run("read", {"path": "pages.pdf"}) for _ in range(3)])
    assert {result.text() for result in results} == {"## Page 1\nFirst page\n\n## Page 2\nSecond page"}
    assert len((await bash(box, "ls -A .agent/views")).split()) == 1  # no temp folder left behind


async def test_a_notebook_reads_as_text(box):
    assert texts(await box.attach("quiz.ipynb", json.dumps(NOTEBOOK).encode())) == [
        "Attached file: uploads/quiz.ipynb\n\nQuiz\n\nName the gas plants give off.\n\n    answer = 'oxygen'"]


@pytest.fixture
def together(monkeypatch, request):
    """Stands in for Together: keeps each request's form fields and answers with a teacher and a
    student, whose answer shares the teacher's second segment."""
    teacher = [("Good", 0.1), ("morning,", 0.4), ("class.", 1.0), ("What", 4.3), ("gas", 4.6), ("do", 4.9),
               ("plants", 5.1), ("give", 5.5), ("off?", 5.8)]
    reply = {"duration": 12.5,
             "segments": [{"start": 0.0, "end": 4.2, "text": " Good morning, class."},
                          {"start": 4.2, "end": 9.8, "text": " What gas do plants give off? Oxygen."}],
             "words": [{"word": word, "start": start, "end": start + 0.3, "speaker_id": "SPEAKER_03"}
                       for word, start in teacher] + [
                       {"word": "Oxygen.", "start": 8.0, "end": 8.6, "speaker_id": "SPEAKER_00"}]}
    received = []

    class Answer(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            form = email.message_from_bytes(f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body,
                                            policy=email.policy.HTTP)
            received.append({"path": self.path, "authorization": self.headers["Authorization"],
                             **{part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                                for part in form.iter_parts()}})
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingTCPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    request.addfinalizer(server.server_close)
    request.addfinalizer(server.shutdown)
    monkeypatch.setattr(transcribe, "TOGETHER_API", f"http://127.0.0.1:{server.server_address[1]}/v1")
    monkeypatch.setenv("TOGETHER_API_KEY", "test")
    return received


async def test_a_recording_is_transcribed_on_the_host(box, together):
    [text] = texts(await box.attach("talk.mp3", b"ID3 recording"))
    assert text == ("Attached file: uploads/talk.mp3 (00:00:12)\n\n[00:00:00] Speaker 1: Good morning, class.\n"
                    "[00:00:04] Speaker 1: What gas do plants give off?\n[00:00:08] Speaker 2: Oxygen.")
    assert together == [{"path": "/v1/audio/transcriptions", "authorization": "Bearer test", "file": b"ID3 recording",
                         "model": b"openai/whisper-large-v3", "response_format": b"verbose_json",
                         "timestamp_granularities": b"segment", "diarize": b"true", "language": b"auto"}]
    assert await bash(box, "cat .agent/views/*/transcript.jsonl") == (
        '{"start": 0.0, "end": 4.2, "speaker": "1", "text": "Good morning, class."}\n'
        '{"start": 4.3, "end": 6.1, "speaker": "1", "text": "What gas do plants give off?"}\n'
        '{"start": 8.0, "end": 8.6, "speaker": "2", "text": "Oxygen."}')
    assert (await box.run("read", {"path": "uploads/talk.mp3"})).content == text.split("\n\n", 1)[1]
    assert len(together) == 1  # read found the view


async def test_a_failed_transcription_is_a_note_and_read_tries_again(box, together, monkeypatch):
    stand_in = transcribe.TOGETHER_API
    monkeypatch.setattr(transcribe, "TOGETHER_API", "http://127.0.0.1:1/v1")  # nothing listens there
    assert texts(await box.attach("talk.mp3", b"ID3 recording")) == [
        "Attached file: uploads/talk.mp3\n\nTogether couldn't transcribe talk.mp3 (ConnectError). Read it to try again."]
    monkeypatch.setattr(transcribe, "TOGETHER_API", stand_in)
    assert (await box.run("read", {"path": "uploads/talk.mp3"})).text().startswith("[00:00:00] Speaker 1: Good morning")


@pytest.mark.skipif(not os.environ.get("TOGETHER_API_KEY"), reason="needs TOGETHER_API_KEY")
async def test_together_transcribes_a_real_recording(box):
    spoken = (pathlib.Path(__file__).parent.parent / "fixtures" / "spoken.mp3").read_bytes()  # "The code is 7."
    [text] = texts(await box.attach("spoken.mp3", spoken))
    assert text.startswith("Attached file: uploads/spoken.mp3 (00:00:01)\n\n[00:00:00] Speaker 1: The code is"), text
    [segment] = [json.loads(line) for line in (await bash(box, "cat .agent/views/*/transcript.jsonl")).splitlines()]
    assert (segment["speaker"], segment["text"].startswith("The code is")) == ("1", True)
