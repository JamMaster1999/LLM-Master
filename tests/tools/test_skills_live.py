"""The shipped skills' scripts, run in the real sandbox image with no network, as a model would run them."""
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from agent import Toolbox
from agent.sandbox import delete

pytestmark = [pytest.mark.live, pytest.mark.asyncio(loop_scope="module")]

SKILLS = Path(__file__).parents[2] / "skills"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def box():
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}", skills=SKILLS)
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


async def bash(box: Toolbox, command: str, *, fails: bool = False) -> str:
    result = await box.run("bash", {"command": command, "timeout": 300})
    assert result.error == fails, result.content
    return result.text()


async def test_the_example_deck_builds_and_passes_the_checker(box):
    out = await bash(box, "mkdir -p deck && cd deck && python /skills/pptx/references/example.py && "
                          "python /skills/pptx/scripts/check.py water-cycle.pptx")
    assert out == "water-cycle.pptx: 6 slides, 13.33 x 7.50 in\nNo problems found."
    assert (await box.run("review_file", {"path": "deck/water-cycle.pptx"})).text().startswith(
        "Reviewed deck/water-cycle.pptx (6 pages).")


async def test_the_checker_reports_text_that_overflows_its_box(box):
    script = ('import sys; sys.path.insert(0, "/skills/pptx/scripts"); from slides import *; deck = new_deck(); '
              'slide = add_slide(deck, "Too much"); add_text(slide, 1, 2, 4, 0.6, "word " * 40); deck.save("full.pptx")')
    out = await bash(box, f"mkdir -p deck && cd deck && python -c '{script}' && python /skills/pptx/scripts/check.py full.pptx",
                     fails=True)
    assert "Slide 1: \"word word word word word word word word…\"'s text overflows its box" in out


async def test_the_checker_measures_chinese_at_full_width(box):
    script = ('import sys; sys.path.insert(0, "/skills/pptx/scripts"); from slides import *; deck = new_deck(); '
              'slide = add_slide(deck, "光合作用"); add_text(slide, 0.8, 1.6, 6, 1, "植物利用阳光、水和二氧化碳制造糖分并释放氧气，'
              '这个过程叫做光合作用，它发生在叶绿体里。", size=24); deck.save("chinese.pptx")')
    out = await bash(box, f"mkdir -p deck && cd deck && python -c '{script}' && python /skills/pptx/scripts/check.py chinese.pptx",
                     fails=True)
    assert "text overflows its box" in out and "no installed font" not in out, out


async def test_a_sideways_phone_photo_goes_in_upright(box):
    script = ('import sys; sys.path.insert(0, "/skills/pptx/scripts"); from slides import *; from PIL import Image; '
              'exif = Image.Exif(); exif[0x0112] = 6; Image.new("RGB", (400, 300), "teal").save("phone.jpg", exif=exif); '
              'deck = new_deck(); slide = add_slide(deck, "Leaf"); '
              'picture = add_picture(slide, "phone.jpg", 1, 1.5, 5, 5, alt="A leaf"); print(picture.height > picture.width)')
    assert await bash(box, f"mkdir -p deck && cd deck && python -c '{script}'") == "True"


async def test_the_checker_reads_a_deck_with_slide_numbers(box):
    script = ('import sys; sys.path.insert(0, "/skills/pptx/scripts"); from slides import *; from lxml import etree; '
              'from pptx.oxml.ns import qn; deck = new_deck(); slide = add_slide(deck, "Numbered"); '
              'add_text(slide, 0.8, 1.6, 11, 4.8, "Body text", size=28); '
              'number = add_text(slide, 12, 6.9, 1, 0.4, "", size=12).text_frame.paragraphs[0]._p; '
              'field = etree.SubElement(number, qn("a:fld"), id="{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}", type="slidenum"); '
              'etree.SubElement(field, qn("a:t")).text = "1"; deck.save("numbered.pptx")')
    out = await bash(box, f"mkdir -p deck && cd deck && python -c '{script}' && python /skills/pptx/scripts/check.py numbered.pptx")
    assert out == "numbered.pptx: 1 slides, 13.33 x 7.50 in\nNo problems found."


async def test_markdown_becomes_a_word_document_with_equations_and_pages(box):
    markdown = ("---\ntitle: Moon phases\n---\n\nName: ______\n\n| Term | Meaning |\n|---|---|\n| Waxing | Growing |\n\n"
                "1. Why does the Moon change shape?\n\n\\lines 2\n\nFind $\\frac{1}{2} + \\frac{1}{4}$.\n\n"
                "\\newpage\n\n## Answer key\n")
    await box.run("write", {"path": "doc/sheet.md", "content": markdown})
    await bash(box, "cd doc && python /skills/docx/scripts/md2docx.py sheet.md sheet.docx")
    out = await bash(box, "cd doc && python -c \"import zipfile; x = zipfile.ZipFile('sheet.docx').read('word/document.xml')"
                          ".decode(); print(x.count('<m:oMath>'), 'lines 2' in x or 'newpage' in x)\"")
    assert out == "1 False"  # an equation Word can edit, and no command left as text
    assert (await box.run("review_file", {"path": "doc/sheet.docx"})).text().startswith("Reviewed doc/sheet.docx (2 pages).")


async def test_recalc_saves_formula_values_and_names_the_errors(box):
    script = ('from openpyxl import Workbook; wb = Workbook(); ws = wb.active; ws.append([90, 80, 0]); '
              'ws["D1"] = "=AVERAGE(A1:B1)"; ws["E1"] = "=A1/C1"; wb.save("book.xlsx")')
    out = await bash(box, f"mkdir -p sheet && cd sheet && python -c '{script}' && python /skills/xlsx/scripts/recalc.py book.xlsx",
                     fails=True)
    assert out.startswith("Recalculated 2 formulas in book.xlsx.\n1 formulas give errors:\nSheet!E1 #DIV/0!: =A1/C1")
    assert (await box.run("read", {"path": "sheet/book.xlsx"})).content == "## Sheet\n90,80,0,85,#DIV/0!"


async def test_recalc_refuses_a_file_that_isnt_xlsx(box):
    out = await bash(box, "mkdir -p sheet && cd sheet && printf 'a,b\\n1,2\\n' > marks.csv && "
                          "python /skills/xlsx/scripts/recalc.py marks.csv", fails=True)
    assert out.startswith("marks.csv isn't an .xlsx file.")
    assert (await box.run("read", {"path": "sheet/marks.csv"})).content == "a,b\n1,2"


async def test_the_example_handout_renders_to_a_two_page_pdf(box):
    await bash(box, "mkdir -p pdf && cd pdf && weasyprint --pdf-tags /skills/pdf/references/example.html handout.pdf")
    out = await bash(box, "cd pdf && pdffonts handout.pdf | tail -n +3 | awk '{print $1}' | cut -d+ -f2 | sort -u")
    assert out == "Carlito\nCarlito-Bold"  # Calibri, through its metric twin, and nothing fell back
    assert (await box.run("review_file", {"path": "pdf/handout.pdf"})).text().startswith("Reviewed pdf/handout.pdf (2 pages).")


async def test_profile_reads_a_one_column_roster_saved_by_excel_on_windows(box):
    await box.sandbox.write("data/names.csv", "name\n\"Muñoz, Ana\"\nPeña\nZoë\n".encode("cp1252"))
    out = await bash(box, "cd data && python /skills/data-analysis/scripts/profile.py names.csv")
    assert "3 rows × 1 columns" in out and "'Muñoz, Ana' ×1" in out and "'Peña' ×1" in out, out


async def test_profile_skips_a_title_above_the_header(box):
    await box.run("write", {"path": "data/period3.csv", "content": "Period 3 Gradebook\n\nstudent,score\nAna,90\nBen,80\n"})
    out = await bash(box, "cd data && python /skills/data-analysis/scripts/profile.py period3.csv")
    assert "skipping 2 lines above the header" in out and "2 rows × 2 columns" in out and "'Ana' ×1" in out, out


async def test_profile_and_the_chart_style_work_on_a_data_file(box):
    await bash(box, "mkdir -p data && cd data && printf 'student,score\\nAna,90\\nBen,\\nCy,70\\n' > scores.csv")
    out = await bash(box, "cd data && python /skills/data-analysis/scripts/profile.py scores.csv")
    assert out.split("\n", 1)[1] == ("3 rows × 2 columns\n  'student': str, 0 missing, 3 distinct; top: 'Ana' ×1, 'Ben' ×1, 'Cy' ×1\n"
                   "  'score': float64, 1 missing, 2 distinct; min 70, median 80, mean 80, max 90")
    await bash(box, "cd data && python -c \"import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; "
                    "plt.style.use('/skills/data-analysis/assets/chart.mplstyle'); fig, ax = plt.subplots(); "
                    "ax.bar(['Ana', 'Cy'], [90, 70]); fig.savefig('chart.png')\"")
    assert (await box.run("review_file", {"path": "data/chart.png"})).text() == "Reviewed data/chart.png."
