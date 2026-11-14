"""Finds the defects a deck's preview hides or makes easy to miss: text that overflows its box, shapes
past the slide's edge, overlapping text, fonts the preview has to substitute, characters no font can
draw, text too small to read from the back of a room, too many words, typed bullet characters, an
empty bottom third, slides without a title, pictures without alt text and empty placeholders.

    python check.py deck.pptx

Text is measured with the fonts LibreOffice renders, so a box it calls full is full in the preview.
Exits 1 when it finds a problem."""
import bisect
import functools
import re
import subprocess
import sys
import unicodedata

from lxml import etree
from PIL import ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn

EMU_PER_PT, EMU_PER_INCH = 12700, 914400
SLACK = EMU_PER_INCH // 100  # a hundredth of an inch past an edge is rounding, not a defect
MIN_SIZE = 18  # points; smaller is hard to read from the back of a room
MIN_FOOTER = 10  # points, for a sources slide and for footers, the text in a slide's bottom 0.8 in
FOOTER = 0.8 * EMU_PER_INCH
EMPTY = 2.0 * EMU_PER_INCH  # space below the lowest shape that makes a slide look unfinished
MAX_WORDS = 60  # on a slide, besides its title and footers; more belongs in the notes
CITATION = re.compile(r"^\s*\[\d+\]|https?://|www\.")  # how a line on a sources slide starts or links
LINE = 1.2  # a line's height per point of size at single spacing, in LibreOffice and PowerPoint
ALIASES = {"calibri": "carlito", "cambria": "caladea", "arial": "liberation sans",
           "times new roman": "liberation serif", "courier new": "liberation mono"}  # metric twins
TITLES = (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)
TYPED_BULLETS = ("•", "●", "▪", "◦", "‣")
MISSING: dict[str, str] = {}  # fonts the deck asks for that aren't installed, and what the preview shows instead


def main(path: str) -> int:
    deck = Presentation(path)
    width, height = deck.slide_width, deck.slide_height
    fonts = theme_fonts(deck.slide_master.part.part_related_by(RT.THEME).blob)
    print(f"{path}: {len(deck.slides)} slides, {inches(width)} x {inches(height)} in")
    problems = []
    for number, slide in enumerate(deck.slides, 1):
        problems += [f"Slide {number}: {problem}" for problem in check_slide(slide, number, width, height, fonts)]
    for family, shown in sorted(MISSING.items()):
        problems.append(f"The deck uses {family}, which isn't installed: the preview shows {shown}, so its line "
                        "breaks are approximate, and other computers may lack it too. A new deck uses Calibri, Cambria, "
                        "Arial or Times New Roman; an uploaded one keeps its own fonts")
    print("\n".join(problems) if problems else "No problems found.")
    return 1 if problems else 0


def check_slide(slide, number, width, height, fonts) -> list[str]:
    problems, boxes, words = [], [], 0
    title = slide.shapes.title
    if title is None or not title.text_frame.text.strip():
        problems.append("no title; give every slide one (screen readers and the outline use it)")
    lines = [line for shape in slide.shapes if shape != title for line in text_of(shape).split("\n") if line.strip()]
    sources = len(lines) > 1 and 2 * sum(bool(CITATION.search(line)) for line in lines) >= len(lines)
    bottoms = [shape.top + shape.height for shape in slide.shapes if shape.top is not None]
    if number > 1 and not sources and bottoms and height - max(bottoms) > EMPTY:
        problems.append(f"the bottom {inches(height - max(bottoms))} in is empty; use the space with larger text "
                        "and boxes")
    for shape in slide.shapes:
        name = describe(shape)
        if shape.left is not None and (min(shape.left, shape.top) < -SLACK or shape.left + shape.width > width + SLACK
                                       or shape.top + shape.height > height + SLACK):
            problems.append(f"{name} runs past the slide's edge")
        if shape.is_placeholder and shape.has_text_frame and not shape.text_frame.text.strip():
            problems.append(f"{name} is an empty placeholder; PowerPoint shows its prompt text. Remove it")
        footer = shape.top is not None and shape.top >= height - FOOTER
        smallest = MIN_FOOTER if sources or footer else MIN_SIZE
        for inner in flattened(shape):  # a group's shapes are checked for their text, not their place
            if inner._element.tag == qn("p:pic"):  # placed pictures and filled picture placeholders
                alt = inner._element.nvPicPr.cNvPr.get("descr", "")
                if not alt or alt.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    problems.append(f"{describe(inner)} has no alt text")
            if inner.has_text_frame and inner.text_frame.text.strip():
                problems += check_frame(inner, describe(inner), fonts, smallest)
                if inner is shape:
                    boxes.append(shape)
            if getattr(inner, "has_table", False):
                problems += check_table(inner, describe(inner), fonts, smallest, height)
            if inner != title and not footer:
                words += len(text_of(inner).split())
    if not sources and words > MAX_WORDS:
        problems.append(f"{words} words of text, too many to read from the back of a room; keep a slide to "
                        f"{MAX_WORDS} and move the detail to the speaker notes or a second slide")
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            across = min(a.left + a.width, b.left + b.width) - max(a.left, b.left)
            down = min(a.top + a.height, b.top + b.height) - max(a.top, b.top)
            if min(across, down) > EMU_PER_INCH / 10:
                problems.append(f"{describe(a)} and {describe(b)} overlap")
    return problems


def check_frame(shape, name, fonts, smallest) -> list[str]:
    frame = shape.text_frame
    body = frame._txBody.find(qn("a:bodyPr"))
    insets = {side: int(body.get(side, default)) for side, default in
              (("lIns", 91440), ("rIns", 91440), ("tIns", 45720), ("bIns", 45720))}
    problems = []
    if body.find(qn("a:normAutofit")) is not None:
        problems.append(f"{name} shrinks text on overflow, which PowerPoint applies only after an edit. "
                        "Size the text to fit instead")
    if any(paragraph.text.lstrip().startswith(TYPED_BULLETS) for paragraph in frame.paragraphs):
        problems.append(f"{name} types bullet characters, which show twice where bullets are on. Use add_bullets")
    paragraphs = [(paragraph, pieces_of(paragraph, shape, fonts)) for paragraph in frame.paragraphs]
    problems += check_fonts(name, [piece for _, pieces in paragraphs for piece in pieces], smallest)
    room = (shape.width - insets["lIns"] - insets["rIns"]) / EMU_PER_PT
    if room <= 0:
        return problems + [f"{name} is too narrow to hold text: widen it to at least 0.5 in"]
    needed = 0.0
    for paragraph, pieces in paragraphs:
        level, pPr = paragraph.level, paragraph._p.pPr
        defaults = pPr.find(qn("a:defRPr")) if pPr is not None else None
        if not pieces:  # a blank line still takes a line's height
            size = font_of(paragraph._p.find(qn("a:endParaRPr")), defaults, shape, level, fonts)[1]
            needed += line_height(paragraph, size)
            continue
        if body.get("wrap") == "none":
            if sum(width_of(font, text) for text, font in pieces) > room:
                problems.append(f"{name} doesn't wrap and its text is wider than the box")
                break
            lines = 1
        else:
            lines = count_lines(pieces, max(room - (int(pPr.get("marL", 0)) / EMU_PER_PT if pPr is not None else 0), 1))
        needed += lines * line_height(paragraph, max(font[1] for _, font in pieces))
    room_h = (shape.height - insets["tIns"] - insets["bIns"]) / EMU_PER_PT
    if needed > room_h + 2:
        problems.append(f"{name}'s text overflows its box: it needs {needed / 72:.2f} in of height and the box "
                        f"has {room_h / 72:.2f} in. Shorten the text, enlarge the box, or split the slide")
    return problems


def line_height(paragraph, size: float) -> float:
    """A line's height in points with its share of the paragraph's spacing: proportional spacing scales
    the font's line, a fixed one replaces it."""
    spacing = paragraph.line_spacing
    line = size * LINE * spacing if isinstance(spacing, float) else spacing.pt if spacing is not None else size * LINE
    before = paragraph.space_before.pt if paragraph.space_before is not None else 0
    after = paragraph.space_after.pt if paragraph.space_after is not None else 0
    return line + before + after


def check_table(frame, name, fonts, smallest, height) -> list[str]:
    """A table's rows grow to fit their text, so its real height is measured here."""
    table, total, pieces = frame.table, 0, []
    for row in table.rows:
        tallest = row.height
        for column, cell in zip(table.columns, row.cells):
            room = (column.width - cell.margin_left - cell.margin_right) / EMU_PER_PT
            needed = 0.0
            for paragraph in cell.text_frame.paragraphs:
                if found := pieces_of(paragraph, None, fonts):
                    pieces += found
                    needed += count_lines(found, room) * max(font[1] for _, font in found) * LINE
            tallest = max(tallest, int(needed * EMU_PER_PT) + cell.margin_top + cell.margin_bottom)
        total += tallest
    problems = check_fonts(name, pieces, smallest)
    if frame.top + total > height + SLACK:
        problems.append(f"{name}'s rows grow to fit their text and run {inches(frame.top + total - height)} in past "
                        "the bottom of the slide")
    return problems


def check_fonts(name, pieces, smallest) -> list[str]:
    """Too-small text, families the preview substitutes, and characters no installed font has."""
    problems = []
    if pieces and (size := min(font[1] for _, font in pieces)) < smallest:
        problems.append(f"{name} has {size:g} pt text, too small to read when projected; use {smallest} pt or more")
    for family in {font[0] for _, font in pieces}:
        if (shown := rendered_family(family)) is not None:
            MISSING[family] = shown
    text = "".join(text for text, _ in pieces)
    if missing := sorted({char for char in text if ord(char) > 127 and not drawable(char)}):
        problems.append(f"{name} has characters no installed font can draw, such as emoji: {' '.join(missing)}. "
                        "The preview shows boxes; use words or shapes")
    return problems


def pieces_of(paragraph, shape, fonts) -> list[tuple[str, tuple]]:
    """The paragraph's text in order as (text, font) pieces: runs, fields such as slide numbers, and
    line breaks."""
    pPr = paragraph._p.pPr
    defaults = pPr.find(qn("a:defRPr")) if pPr is not None else None
    font, pieces = font_of(None, defaults, shape, paragraph.level, fonts), []
    for child in paragraph._p:
        if child.tag in (qn("a:r"), qn("a:fld")):
            font = font_of(child.find(qn("a:rPr")), defaults, shape, paragraph.level, fonts)
            pieces.append((child.findtext(qn("a:t")) or "", font))
        elif child.tag == qn("a:br"):
            pieces.append(("\n", font))
    return [(text, font) for text, font in pieces if text]


def font_of(rPr, defaults, shape, level, fonts) -> tuple[str, float, bool, bool]:
    """Family, size in points, bold and italic: the run's own, then its paragraph's, then the list
    styles of the shape, its layout and the master, then the defaults."""
    chain = [props for props in (rPr, defaults, *inherited(shape, level)) if props is not None]
    def first(attribute):
        return next((props.get(attribute) for props in chain if props.get(attribute) is not None), None)
    latin = next((props.find(qn("a:latin")) for props in chain if props.find(qn("a:latin")) is not None), None)
    family = latin.get("typeface") if latin is not None else None
    if family in (None, "+mj-lt", "+mn-lt"):  # the theme's heading or body font
        major = family == "+mj-lt" or (family is None and shape is not None and is_title(shape))
        family = fonts["major" if major else "minor"]
    size = int(first("sz")) / 100 if first("sz") else 18.0  # the default text style's size
    return family, size, first("b") in ("1", "true"), first("i") in ("1", "true")


def inherited(shape, level: int) -> tuple:
    """The level's defaults from the shape's list styles, its layout placeholder's, and the master's
    text styles, nearest first."""
    tag = qn(f"a:lvl{level + 1}pPr")
    owners = []
    if shape is not None:
        owners.append(shape.text_frame._txBody)
        if shape.is_placeholder:
            layout = shape.part.slide_layout.placeholders.get(idx=shape.placeholder_format.idx)
            if layout is not None:
                owners.append(layout.text_frame._txBody)
            styles = shape.part.slide_layout.slide_master.element.find(qn("p:txStyles"))
            style = "p:titleStyle" if is_title(shape) else "p:bodyStyle" if shape.placeholder_format.type in (
                PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT, PP_PLACEHOLDER.SUBTITLE) else "p:otherStyle"
            if styles is not None and styles.find(qn(style)) is not None:
                owners.append(styles.find(qn(style)))
    return tuple(found for owner in owners if (found := owner.find(f".//{tag}/{qn('a:defRPr')}")) is not None)


def count_lines(pieces, room: float) -> int:
    """Greedy word wrap, as LibreOffice and PowerPoint do it. A word may run across pieces."""
    tokens = []  # ("word" | "space" | "break", width)
    for text, font in pieces:
        for part in re.split(r"( |\n)", text):
            if part == "\n":
                tokens.append(("break", 0.0))
            elif part == " ":
                tokens.append(("space", width_of(font, " ")))
            elif part and tokens and tokens[-1][0] == "word":
                tokens[-1] = ("word", tokens[-1][1] + width_of(font, part))
            elif part:
                tokens.append(("word", width_of(font, part)))
    lines, used, spaces = 1, 0.0, 0.0
    for kind, width in tokens:
        if kind == "break":
            lines, used, spaces = lines + 1, 0.0, 0.0
        elif kind == "space":
            spaces += width
        else:
            if used and used + spaces + width > room:
                lines, used = lines + 1, 0.0
            if not used and width > room:  # a word longer than the line breaks inside itself
                extra = int(width // room)
                lines, width = lines + extra, width - extra * room
            used, spaces = used + (spaces if used else 0) + width, 0.0
    return lines


def flattened(shape):
    yield shape
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for inner in shape.shapes:
            yield from flattened(inner)


def text_of(shape) -> str:
    if shape.has_text_frame:
        return shape.text_frame.text
    if getattr(shape, "has_table", False) and shape.has_table:
        return " ".join(cell.text for row in shape.table.rows for cell in row.cells)
    return ""


def is_title(shape) -> bool:
    return shape.is_placeholder and shape.placeholder_format.type in TITLES


@functools.cache
def theme_fonts(blob: bytes) -> dict:
    theme = etree.fromstring(blob)
    return {kind: theme.find(f".//{qn(f'a:{kind}Font')}/{qn('a:latin')}").get("typeface") for kind in ("major", "minor")}


def fontconfig(*args: str) -> str:
    return subprocess.run(["fc-match" if args[0] == "match" else "fc-list", *args[1:]], capture_output=True,
                          text=True).stdout


@functools.cache
def rendered_family(family: str) -> str | None:
    """None when the family, or its metric twin, is installed; else what fontconfig shows instead."""
    shown = fontconfig("match", "-f", "%{family[0]}", family)
    return None if shown.lower() in (family.lower(), ALIASES.get(family.lower())) else shown


@functools.cache
def coverage() -> tuple[list[int], list[int]]:
    """Every installed font's characters, as merged ranges: the starts, and the ends."""
    ranges = sorted((int(low, 16), int(high or low, 16)) for line in fontconfig("list", "--format", "%{charset}\n").split()
                    for low, _, high in [line.partition("-")])
    merged = []
    for low, high in ranges:
        if merged and low <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return [low for low, _ in merged], [high for _, high in merged]


def drawable(char: str) -> bool:
    """Whether some installed font has the character. LibreOffice falls back to any font that does."""
    if unicodedata.category(char) in ("Mn", "Me", "Cf", "Zs", "Zl", "Zp"):  # marks and joiners draw nothing
        return True
    starts, ends = coverage()
    i = bisect.bisect_right(starts, ord(char)) - 1
    return i >= 0 and ord(char) <= ends[i]


@functools.cache
def loaded(family: str, bold: bool, italic: bool):
    file = fontconfig("match", "-f", "%{file}", f"{family}{':bold' * bold}{':italic' * italic}")
    return ImageFont.truetype(file, 100)  # 100 px: widths to a hundredth of an em


def width_of(font, text: str) -> float:
    """Points. Chinese, Japanese and Korean characters take a full em, in whatever font draws them."""
    family, size, bold, italic = font
    wide = sum(unicodedata.east_asian_width(char) in "WF" for char in text)
    if wide:
        text = "".join(char for char in text if unicodedata.east_asian_width(char) not in "WF")
    return (loaded(family, bold, italic).getlength(text) / 100 + wide) * size


def describe(shape) -> str:
    text = text_of(shape).strip().split("\n")[0]
    return f'"{text[:40].rstrip()}…"' if len(text) > 40 else f'"{text}"' if text else shape.name


def inches(emu) -> str:
    return f"{emu / EMU_PER_INCH:.2f}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
