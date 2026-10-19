"""Markdown to Word through pandoc, in a clean house style: Calibri, dark blue headings, ruled tables
with a shaded header row, page numbers in the footer, and 1-inch margins on Letter paper.

    python md2docx.py notes.md notes.docx [--a4] [--landscape] [--size 14]

Two commands, each alone on its line: \\newpage starts a new page, and \\lines 3 draws three ruled
lines to write on. Images take a width in inches, ![A leaf](leaf.png){width=3in}, with paths relative
to the Markdown file."""
import argparse
import os
import re
import subprocess
import sys
import tempfile

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Mm, Pt, RGBColor

FONT = "Calibri"
INK, ACCENT, MUTED, RULE, SHADE = "1F2937", "1F4E79", "6B7280", "9CA3AF", "E8EEF5"
# style: size, bold, color, space before, space after
STYLES = {"Normal": (11, False, INK, 0, 0), "Body Text": (11, False, INK, 0, 8), "First Paragraph": (11, False, INK, 0, 8),
          "Compact": (11, False, INK, 0, 3), "Title": (24, True, ACCENT, 0, 4), "Subtitle": (14, False, MUTED, 0, 12),
          "Heading 1": (16, True, ACCENT, 18, 6), "Heading 2": (13, True, ACCENT, 12, 4),
          "Heading 3": (11, True, INK, 10, 2), "Heading 4": (11, True, INK, 8, 2), "Heading 5": (11, True, INK, 8, 2),
          "Heading 6": (11, True, INK, 8, 2), "Author": (11, False, MUTED, 0, 0), "Date": (11, False, MUTED, 0, 12),
          "Footnote Text": (9, False, INK, 0, 0)}
PAGE_BREAK = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'
# writing lines as a table's row borders: Word merges the borders of like paragraphs into one box
LINE = ('<w:tr><w:trPr><w:trHeight w:val="560" w:hRule="exact"/></w:trPr>'
        '<w:tc><w:tcPr><w:tcW w:w="5000" w:type="pct"/></w:tcPr><w:p/></w:tc></w:tr>')
LINES = ('```{{=openxml}}\n<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders>'
         f'<w:insideH w:val="single" w:sz="4" w:color="{RULE}"/><w:bottom w:val="single" w:sz="4" w:color="{RULE}"/>'
         '</w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w="9360"/></w:tblGrid>{rows}</w:tbl><w:p/>\n```')


def reference(path: str, a4: bool, landscape: bool, size: float) -> None:
    """pandoc's own reference document, restyled, with body text at `size` points and the rest in scale."""
    subprocess.run(["pandoc", "-o", path, "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(path)
    styles = {style.name: style for style in doc.styles}  # by name: pandoc capitalizes "Heading 1", which python-docx's lookup misses
    scale = size / STYLES["Normal"][0]
    for name, (points, bold, color, before, after) in STYLES.items():
        style = styles[name]
        style.font.name, style.font.size, style.font.bold = FONT, Pt(round(points * scale * 2) / 2), bold
        style.font.italic = False
        style.font.color.rgb = RGBColor.from_string(color)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        for theme in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):  # they outrank the name
            style.element.rPr.rFonts.attrib.pop(qn(theme), None)
        style.paragraph_format.space_before, style.paragraph_format.space_after = Pt(before), Pt(after)
        if name in ("Body Text", "First Paragraph", "Compact"):
            style.paragraph_format.line_spacing = 1.15
    _rule_tables(styles["Table"].element)
    section = doc.sections[0]
    width, height = (Mm(210), Mm(297)) if a4 else (Inches(8.5), Inches(11))
    if landscape:
        section.orientation, width, height = WD_ORIENT.LANDSCAPE, height, width
    section.page_width, section.page_height = width, height
    section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = Inches(1)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _page_number(footer)
    doc.save(path)


def _rule_tables(style) -> None:
    """Thin gray rules around every cell, and a bold, shaded header row, for the tables pandoc makes.
    It stops being the default table style, so writing lines and python-docx tables don't take it."""
    style.attrib.pop(qn("w:default"), None)
    for old in [style.find(qn("w:tblPr")), *style.findall(qn("w:tblStylePr"))]:
        if old is not None:
            style.remove(old)
    edges = "".join(f'<w:{edge} w:val="single" w:sz="4" w:color="{RULE}"/>'
                    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"))
    style.append(parse_xml(f'<w:tblPr {nsdecls("w")}><w:tblInd w:w="0" w:type="dxa"/><w:tblBorders>{edges}'
                           '</w:tblBorders><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
                           '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr>'))
    style.append(parse_xml(f'<w:tblStylePr {nsdecls("w")} w:type="firstRow"><w:rPr><w:b/></w:rPr><w:tcPr>'
                           f'<w:shd w:val="clear" w:color="auto" w:fill="{SHADE}"/></w:tcPr></w:tblStylePr>'))


def _page_number(paragraph) -> None:
    paragraph._p.append(parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"><w:r><w:rPr><w:color w:val="{MUTED}"/>'
                                  '<w:sz w:val="18"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("markdown")
    parser.add_argument("docx")
    parser.add_argument("--a4", action="store_true", help="A4 paper instead of Letter")
    parser.add_argument("--landscape", action="store_true")
    parser.add_argument("--size", type=float, default=11, help="body text in points; 14 or more for young readers")
    args = parser.parse_args()
    text = re.sub(r"(?m)^\\newpage\s*$", PAGE_BREAK, open(args.markdown, encoding="utf-8").read())
    text = re.sub(r"(?m)^[ \t]*\\lines (\d+)\s*$", lambda m: LINES.format(rows=LINE * int(m[1])), text)
    with tempfile.TemporaryDirectory() as work:
        ref = os.path.join(work, "reference.docx")
        reference(ref, args.a4, args.landscape, args.size)
        # implicit figures would print each image's alt text under it as a caption
        done = subprocess.run(["pandoc", "-f", "markdown-implicit_figures+lists_without_preceding_blankline", "-t",
                               "docx", "--reference-doc", ref, "--resource-path",
                               os.path.dirname(os.path.abspath(args.markdown)), "-o", args.docx],
                              input=text.encode(), capture_output=True, check=True)
    warnings = done.stderr.decode()
    print(warnings, end="")
    if "Could not fetch resource" in warnings:  # pandoc swaps a missing image for its alt text and carries on
        sys.exit("An image is missing: give its path in the sandbox, relative to the Markdown file.")
    print(f"Wrote {args.docx}")


if __name__ == "__main__":
    main()
