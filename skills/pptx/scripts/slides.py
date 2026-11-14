"""Helpers over python-pptx for decks that look right in PowerPoint and in LibreOffice's preview.

Sizes are inches on a 13.333 x 7.5 slide, colors are hex strings like "1F4E3D", and text may mark
bold words with **double stars**. Every helper returns the python-pptx object it made, so anything
python-pptx can do still applies."""
import io
import os
import re

import pillow_heif
from lxml import etree
from PIL import Image, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

WIDTH, HEIGHT = 13.333, 7.5
SHAPES = {"rect": MSO_SHAPE.RECTANGLE, "rounded": MSO_SHAPE.ROUNDED_RECTANGLE, "oval": MSO_SHAPE.OVAL,
          "arrow": MSO_SHAPE.RIGHT_ARROW, "chevron": MSO_SHAPE.CHEVRON, "pentagon": MSO_SHAPE.PENTAGON}
ALIGN = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}
ANCHOR = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}
PLACED = ("PNG", "JPEG", "GIF")  # what python-pptx and PowerPoint both take as they are
pillow_heif.register_heif_opener()  # iPhone photos


def new_deck() -> Presentation:
    """A 16:9 deck. python-pptx's own default is 4:3, with its layouts drawn for that width."""
    deck = Presentation()
    scale = Inches(WIDTH) / deck.slide_width
    # layouts first: a layout placeholder without a position of its own reads the master's
    for owner in [*deck.slide_layouts, deck.slide_master]:
        for shape in owner.placeholders:
            left, top, width, height = shape.left, shape.top, shape.width, shape.height
            shape.left, shape.top, shape.width, shape.height = int(left * scale), top, int(width * scale), height
    deck.slide_width, deck.slide_height = Inches(WIDTH), Inches(HEIGHT)
    return deck


def add_slide(deck, title: str, *, bg: str | None = None, color: str = "1F2937", size: int = 36,
              box: tuple = (0.6, 0.35, 12.1, 1.0), align: str = "left", font: str = "Calibri"):
    """A slide whose title sits in the real title placeholder, so it shows in the outline and screen
    readers find it. `box` is the title's (x, y, w, h)."""
    slide = deck.slides.add_slide(deck.slide_layouts[5])  # Title Only
    if bg:
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _rgb(bg)
    title_shape = slide.shapes.title
    title_shape.left, title_shape.top, title_shape.width, title_shape.height = (Inches(v) for v in box)
    _fill(title_shape.text_frame, [title], size=size, color=color, bold=True, align=align, font=font,
          anchor="middle")
    return slide


def add_text(slide, x, y, w, h, text: str | list[str], *, size: int = 24, color: str = "1F2937",
             bold: bool = False, italic: bool = False, align: str = "left", anchor: str = "top",
             font: str = "Calibri", spacing: float = 0.3):
    """A wrapping text box. A list gives one paragraph per item, `spacing` (in lines) apart."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    _fill(box.text_frame, [text] if isinstance(text, str) else text, size=size, color=color, bold=bold,
          italic=italic, align=align, font=font, anchor=anchor, spacing=spacing)
    return box


def add_bullets(slide, x, y, w, h, items: list, *, size: int = 24, color: str = "1F2937",
                bullet_color: str | None = None, numbered: bool = False, font: str = "Calibri",
                spacing: float = 0.4):
    """Real bullets, not typed "•" characters. A list nested in `items` holds the sub-points of the
    item before it. `numbered` numbers the top level."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    flat = list(_levels(items))
    _fill(box.text_frame, [text for text, _ in flat], size=size, color=color, font=font, spacing=spacing)
    for paragraph, (_, level) in zip(box.text_frame.paragraphs, flat):
        paragraph.level = level  # the outline and screen readers see the nesting, and numbering restarts per level
        step = 0.4 * size / 24  # the indent grows with the text
        pPr = paragraph._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(step * (level + 1))))
        pPr.set("indent", str(-Inches(step)))
        if bullet_color:
            pPr.append(_element("a:buClr", _element("a:srgbClr", val=bullet_color.lstrip("#"))))
        if numbered and level == 0:
            pPr.append(_element("a:buAutoNum", type="arabicPeriod"))
        else:
            pPr.append(_element("a:buFont", typeface="Arial"))
            pPr.append(_element("a:buChar", char="•" if level == 0 else "–"))
        if level:
            for run in paragraph.runs:
                run.font.size = Pt(round(size * 0.85))
    return box


def add_box(slide, x, y, w, h, *, fill: str = "F3F4F6", line: str | None = None, shape: str = "rounded",
            text: str | list[str] | None = None, size: int = 20, color: str = "1F2937", bold: bool = False,
            align: str = "center", anchor: str = "middle", font: str = "Calibri"):
    """A filled shape (rect, rounded, oval, arrow, chevron, pentagon), with optional text inside.
    Use it for cards, labels and the steps of a process."""
    box = slide.shapes.add_shape(SHAPES[shape], Inches(x), Inches(y), Inches(w), Inches(h))
    box._element.remove(box._element.find(qn("p:style")))  # its theme shadow, which LibreOffice draws regardless
    box.fill.solid()
    box.fill.fore_color.rgb = _rgb(fill)
    if line:
        box.line.color.rgb = _rgb(line)
        box.line.width = Pt(1.5)
    else:
        box.line.fill.background()
    if shape == "rounded":
        box.adjustments[0] = 0.1
    if text is not None:
        _fill(box.text_frame, [text] if isinstance(text, str) else text, size=size, color=color, bold=bold,
              align=align, font=font, anchor=anchor)
    return box


def add_arrow(slide, x1, y1, x2, y2, *, color: str = "6B7280", width: float = 2.5):
    """A straight line from (x1, y1) to (x2, y2) with an arrowhead at the end."""
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = _rgb(color)
    line.line.width = Pt(width)
    line.line._get_or_add_ln().append(_element("a:tailEnd", type="triangle", w="med", len="med"))
    return line


def add_picture(slide, path: str, x, y, w, h, *, alt: str):
    """The picture scaled to fit inside the box without distortion, centered in it. `alt` describes it
    for screen readers. A picture over 1 MB goes in at most 1600 pixels wide, so a deck of generated
    images stays small; one PowerPoint can't take, such as WebP or HEIC, is converted; and a phone photo
    is turned upright. Each of those goes in as JPEG, or PNG when it's transparent."""
    with Image.open(path) as image:
        turned = image.getexif().get(0x0112, 1) != 1  # a phone photo saved sideways, with a note saying so
        upright = ImageOps.exif_transpose(image)
        ratio = upright.width / upright.height
        source = path
        if os.path.getsize(path) > 1_000_000 or image.format not in PLACED or turned:
            image = upright
            image.thumbnail((1600, 1600))
            source, transparent = io.BytesIO(), image.mode in ("RGBA", "LA", "P")
            image = image if transparent or image.mode == "RGB" else image.convert("RGB")
            image.save(source, "PNG" if transparent else "JPEG", **({} if transparent else {"quality": 85}))
            source.seek(0)
    fit_w, fit_h = (w, w / ratio) if w / h < ratio else (h * ratio, h)
    picture = slide.shapes.add_picture(source, Inches(x + (w - fit_w) / 2), Inches(y + (h - fit_h) / 2),
                                       Inches(fit_w), Inches(fit_h))
    picture._element.nvPicPr.cNvPr.set("descr", alt)
    return picture


def set_notes(slide, text: str) -> None:
    """Speaker notes: what the presenter says, asks and answers. They never show on the slide."""
    slide.notes_slide.notes_text_frame.text = text


def _fill(frame, paragraphs: list[str], *, size, color, font, bold=False, italic=False, align="left",
          anchor="top", spacing=0.3) -> None:
    frame.word_wrap = True  # python-pptx's text boxes default to one endless line
    frame.auto_size = MSO_AUTO_SIZE.NONE  # LibreOffice would grow the box and PowerPoint wouldn't
    frame.vertical_anchor = ANCHOR[anchor]
    frame.text = ""
    for i, text in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = ALIGN[align]
        if i < len(paragraphs) - 1:
            paragraph.space_after = Pt(round(size * spacing))
        for j, piece in enumerate(re.split(r"\*\*", text)):
            if piece:
                run = paragraph.add_run()
                run.text = piece
                run.font.size, run.font.name = Pt(size), font
                run.font.bold, run.font.italic = bold or j % 2 == 1, italic
                run.font.color.rgb = _rgb(color)


def _levels(items, level=0):
    for item in items:
        if isinstance(item, list):
            yield from _levels(item, level + 1)
        else:
            yield item, level


def _rgb(color: str) -> RGBColor:
    return RGBColor.from_string(color.lstrip("#"))


def _element(tag: str, child=None, **attributes):
    element = etree.Element(qn(tag), {key: str(value) for key, value in attributes.items()})
    if child is not None:
        element.append(child)
    return element
