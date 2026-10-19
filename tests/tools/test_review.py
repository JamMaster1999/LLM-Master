"""review_file and deliver_file's gate through Toolbox.run, on real Modal."""
import base64
import io
import uuid

import pytest
from PIL import Image

from agent import Toolbox
from agent.sandbox import delete

pytestmark = pytest.mark.live

DOCUMENT = ("printf '%s' '<p>{first}</p><p style=\"page-break-before: always\">Second page</p>' > pages.html && "
            "soffice --headless --infilter='HTML (StarWriter)' --convert-to docx pages.html >/dev/null && "
            "soffice --headless --convert-to pdf pages.docx >/dev/null")


@pytest.fixture
async def box():
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}")
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


async def bash(box: Toolbox, command: str) -> None:
    result = await box.run("bash", {"command": command})
    assert not result.error, result.content


async def test_review_file_shows_every_page_and_its_first_line(box):
    await bash(box, DOCUMENT.format(first="First page") + " && echo hi > notes.txt")
    summary, sheet = (await box.run("review_file", {"path": "pages.docx"})).content
    assert summary["text"] == ("Reviewed pages.docx (2 pages). The first line of each page:\n1. First page\n"
                               "2. Second page\n\nTo see a page in detail, read it with pages.")
    assert max(Image.open(io.BytesIO(base64.b64decode(sheet["data"]))).size) <= 1568
    refused = await box.run("review_file", {"path": "notes.txt"})
    assert refused.error and refused.text().startswith("review_file checks PDFs, documents, slides and images.")


async def test_deliver_file_sends_a_document_once_its_current_version_is_reviewed(box):
    await bash(box, DOCUMENT.format(first="First page"))
    refused = await box.run("deliver_file", {"path": "pages.docx"})
    assert (refused.error, refused.text()) == (
        True, "pages.docx has not been reviewed yet. Review it with review_file before sending it.")
    await box.run("review_file", {"path": "pages.docx"})
    sent = await box.run("deliver_file", {"path": "pages.docx"})
    assert sent.content == "Sent pages.docx and pages.pdf to the user."
    assert [(file.name, file.mime, file.data[:4]) for file in sent.files] == [
        ("pages.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"PK\x03\x04"),
        ("pages.pdf", "application/pdf", b"%PDF")]
    await bash(box, DOCUMENT.format(first="Page one"))
    changed = await box.run("deliver_file", {"path": "pages.docx"})
    assert (changed.error, changed.text()) == (
        True, "pages.docx has changed since it was reviewed. Review it again before sending it.")


async def test_an_image_or_a_pdf_goes_alone_once_reviewed(box):
    await bash(box, DOCUMENT.format(first="First page") +
               " && python -c \"from PIL import Image; Image.new('RGB', (40, 30), 'red').save('chart.png')\"")
    assert (await box.run("deliver_file", {"path": "chart.png"})).error
    assert [part["type"] for part in (await box.run("review_file", {"path": "chart.png"})).content] == ["text", "image"]
    await box.run("review_file", {"path": "pages.pdf"})
    for path in ("chart.png", "pages.pdf"):
        assert [file.name for file in (await box.run("deliver_file", {"path": path})).files] == [path]
