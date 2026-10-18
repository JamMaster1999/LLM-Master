"""Every sandbox and web tool through Toolbox.run, on real Modal and Exa."""
import asyncio
import base64
import uuid

import pytest

from agent import Toolbox
from agent.sandbox import delete

pytestmark = pytest.mark.live

WIKI = "https://en.wikipedia.org/wiki/Photosynthesis"
TWO_PAGES = '<p>First page</p><p style="page-break-before: always">Second page</p>'


@pytest.fixture
async def box():
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}", urls=[WIKI])
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


async def run(box: Toolbox, name: str, **arguments):
    return await box.run(name, arguments)


def parts(result) -> list[dict]:
    assert isinstance(result.content, list), result.content
    return result.content


async def test_write_edit_read_round_trip(box):
    assert (await run(box, "write", path="notes/plan.txt", content="one\ntwo\n")).content == (
        "Successfully wrote to notes/plan.txt")
    edited = await run(box, "edit", path="notes/plan.txt", edits=[{"oldText": "two", "newText": "2"}])
    assert edited.content == "Successfully replaced 1 block(s) in notes/plan.txt."
    assert (await run(box, "read", path="/workspace/notes/plan.txt")).content == "one\n2"


async def test_a_refused_edit_is_an_error_result(box):
    await run(box, "write", path="plan.txt", content="one\n")
    result = await run(box, "edit", path="plan.txt", edits=[{"oldText": "three", "newText": "3"}])
    assert result.error and result.text().startswith("Could not find the exact text in plan.txt.")


async def test_parallel_edits_to_one_file_both_land(box):
    await run(box, "write", path="plan.txt", content="one\ntwo\n")
    await asyncio.gather(run(box, "edit", path="plan.txt", edits=[{"oldText": "one", "newText": "1"}]),
                         run(box, "edit", path="plan.txt", edits=[{"oldText": "two", "newText": "2"}]))
    assert (await run(box, "read", path="plan.txt")).content == "1\n2"


async def test_read_says_what_is_wrong_with_a_path(box):
    await run(box, "bash", command=r"mkdir -p folder && printf 'PK\003\004\377\376' > blob.bin")
    for path, message in [("missing.txt", "File not found: missing.txt"), ("folder", "folder is a directory."),
                          ("talk.mp3", "read can't return audio or video."), ("blob.bin", "blob.bin is a binary file")]:
        result = await run(box, "read", path=path)
        assert result.error and result.text().startswith(message), result.content


async def test_read_renders_pdfs_and_documents_as_page_images(box):
    await run(box, "write", path="pages.html", content=TWO_PAGES)
    await run(box, "bash", command='soffice --headless --infilter="HTML (StarWriter)" --convert-to docx pages.html && '
                                   "soffice --headless --convert-to pdf pages.html")
    header, *pages = parts(await run(box, "read", path="pages.docx", limit=1))
    assert header["text"] == "Rendered pages.docx [pages 1-1 of 2] Use offset=2 to continue."
    assert [page["mime"] for page in pages] == ["image/jpeg"]
    header, page = parts(await run(box, "read", path="pages.pdf", offset=2))
    assert header["text"] == "Rendered pages.pdf [pages 2-2 of 2]"
    assert base64.b64decode(page["data"])[:2] == b"\xff\xd8"


async def test_read_scales_large_images_and_passes_small_ones(box):
    await run(box, "bash", command="python -c \"from PIL import Image; Image.new('RGB', (3000, 1000)).save('big.png'); "
                                   "Image.new('RGB', (40, 40)).save('small.png')\"")
    big, small = parts(await run(box, "read", path="big.png")), parts(await run(box, "read", path="small.png"))
    assert big[0]["text"] == "Read image file [image/jpeg]"
    assert base64.b64decode(big[1]["data"])[:2] == b"\xff\xd8"
    assert small[1]["mime"] == "image/png"


async def test_read_turns_spreadsheets_into_csv(box):
    await run(box, "write", path="small.csv", content="name,score\nAda,9\n")
    assert (await run(box, "read", path="small.csv")).content == "## small.csv\nname,score\nAda,9"
    await run(box, "bash", command="python -c \"import openpyxl; book = openpyxl.Workbook(); sheet = book.active; "
                                   "sheet.title = 'Scores'; sheet.append([1]); sheet.append([2]); sheet.append(['=SUM(A1:A2)']); "
                                   "book.save('built.xlsx')\"")
    assert (await run(box, "read", path="built.xlsx")).content == "## Scores\n1\n2\n=SUM(A1:A2)"
    await run(box, "write", path="big.csv", content="n\n" + "\n".join(str(i) for i in range(10_000)))
    preview = (await run(box, "read", path="big.csv")).text()
    assert "[10001 rows x 1 columns; showing the first 20]" in preview
    assert preview.endswith("It stays at /workspace/big.csv; use bash or run_program to analyze it.]")


async def test_bash_reports_output_exit_code_and_timeout(box):
    failed = await run(box, "bash", command="echo hi; exit 3")
    assert (failed.content, failed.error) == ("hi\n\nCommand exited with code 3", True)
    slow = await run(box, "bash", command="sleep 5", timeout=1)
    assert slow.content == "(no output)\n\nCommand timed out after 1 seconds"
    refused = await run(box, "bash", command="true", timeout=601)
    assert refused.error and refused.content == "timeout must be more than 0 and at most 600 seconds."


async def test_bash_cuts_long_output_and_keeps_the_rest_for_read(box):
    result = await run(box, "bash", command="seq 5000")
    shown, notice = result.text().rsplit("\n\n", 1)
    assert shown.splitlines()[0] == "3001"
    assert notice.startswith("[Showing lines 3001-5000 of 5000. Full output: .agent/bash-")
    log = notice.removeprefix("[Showing lines 3001-5000 of 5000. Full output: ").removesuffix("]")
    assert (await run(box, "read", path=log, limit=2)).content == "1\n2\n\n[4998 more lines in file. Use offset=3 to continue.]"


async def test_run_program_calls_host_tools_and_the_host_refuses_the_rest(box):
    code = """
with open("report.txt", "w") as f:
    f.write("Quiz report")
print(await deliver_file({"path": "report.txt"}))
import __main__  # the runner: a program can forge any call through it
try:
    await __main__.tool("bash")({"command": "ls"})
except ToolError as error:
    print(error)
"""
    result = await run(box, "run_program", code=code)
    assert result.content == "Sent report.txt to the user.\nbash can't be called from a program."
    assert [(file.name, file.data) for file in result.files] == [("report.txt", b"Quiz report")]


async def test_run_program_shows_the_programs_traceback(box):
    result = await run(box, "run_program", code="x = 1\nraise ValueError('no grades yet')\n")
    assert result.error
    assert 'line 2, in <module>\n    raise ValueError' in result.content
    assert result.text().endswith("ValueError: no grades yet\n\nProgram exited with code 1")


async def test_deliver_file_hands_the_bytes_to_the_user(box):
    await run(box, "write", path="out/notes.md", content="# Notes")
    result = await run(box, "deliver_file", path="out/notes.md")
    assert result.content == "Sent notes.md to the user."
    assert [(f.name, f.mime, f.data) for f in result.files] == [("notes.md", "text/markdown", b"# Notes")]


async def test_web_search_numbers_sources_after_the_saved_ones():
    saved = [{"type": "source-url", "sourceId": "1", "url": "https://example.com", "title": "Example"}]
    box = Toolbox("unused", sources=saved)
    try:
        result = await run(box, "web_search", query="a grade 9 lesson plan on the light reactions of photosynthesis",
                           objective="Lesson plans a high school biology teacher could use")
        again = await run(box, "web_search", query="a grade 9 lesson plan on the light reactions of photosynthesis",
                          objective="Lesson plans a high school biology teacher could use")
    finally:
        await box.aclose()
    assert result.text().startswith("[2] Title: ")
    assert [part["sourceId"] for part in result.sources] == [str(n) for n in range(2, 2 + len(result.sources))]
    first = {part["url"]: part["sourceId"] for part in result.sources}
    assert all(first.get(part["url"], part["sourceId"]) == part["sourceId"] for part in again.sources)


async def test_fetch_url_reads_seen_urls_and_saves_long_pages(box):
    unseen = await run(box, "fetch_url", urls=["https://example.com/made-up"])
    assert unseen.error and "Not seen: https://example.com/made-up" in unseen.content
    page = await run(box, "fetch_url", urls=[WIKI])
    assert page.text().startswith(f"[1] Title: Photosynthesis\nURL: {WIKI}")
    assert "The whole page is at .agent/page-1.txt; read it with offset and limit.]" in page.content
    assert (await run(box, "read", path=".agent/page-1.txt", limit=1)).text().startswith("Photosynthesis")
