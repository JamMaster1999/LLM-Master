# Changing an uploaded deck

Work on a copy and save under a new name, such as `uploads/unit3.pptx` → `unit3-revised.pptx`. A
legacy `.ppt` converts first: `soffice --headless --convert-to pptx --outdir . uploads/old.ppt`.

## See what's there

```python
from pptx import Presentation
from pptx.oxml.ns import qn

deck = Presentation("uploads/unit3.pptx")
print(f"{deck.slide_width / 914400:.2f} x {deck.slide_height / 914400:.2f} in")
for number, slide in enumerate(deck.slides, 1):
    print(number, slide.slide_layout.name)
    for shape in slide.shapes:
        where = f"placeholder {shape.placeholder_format.idx}" if shape.is_placeholder else shape.shape_type
        text = shape.text_frame.text[:60].replace("\n", " / ") if shape.has_text_frame else ""
        print(f"   {shape.name} ({where}): {text}")
```

Read pages with `read(path, pages=...)` to see how each layout looks.

## Change text and keep its formatting

Setting `text_frame.text` or `paragraph.text` drops the run's font, size and color. Keep the first
run, and remove the other runs and the line breaks:

```python
def replace_text(paragraph, text):
    first = (paragraph.runs or [paragraph.add_run()])[0]
    for child in list(paragraph._p):
        if child.tag in (qn("a:r"), qn("a:br"), qn("a:fld")) and child is not first._r:
            paragraph._p.remove(child)
    first.text = text
```

## Add, delete and reorder slides

- **Add** with the deck's own layout, then fill its placeholders by index:
  `slide = deck.slides.add_slide(deck.slide_layouts.get_by_name("Title and Content"))`, then
  `slide.placeholders[1].text_frame.text = "..."`. Delete placeholders you leave empty:
  `shape._element.getparent().remove(shape._element)`.
- **Delete** slide `i` (0-based):

  ```python
  ids = deck.slides._sldIdLst
  slide_id = ids[i]
  deck.part.drop_rel(slide_id.rId)
  ids.remove(slide_id)
  ```

- **Reorder** by moving the entry: `ids.insert(new_index, ids[old_index])`.
- **Duplicating** a slide isn't supported by python-pptx. Add a slide with the same layout and build
  its content again.

## Check

Run `python /skills/pptx/scripts/check.py` and review the pages as for a new deck. The checker also reports problems the
upload already had. Fix the ones on slides you changed, and mention the others to the user.
