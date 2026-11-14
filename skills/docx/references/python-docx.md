# python-docx: finishing and changing documents

Open the file, change it, and save under a new name. Then review it as for a new document.

```python
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

doc = Document("uploads/syllabus.docx")
```

## See what's there

```python
for i, paragraph in enumerate(doc.paragraphs):
    print(i, paragraph.style.name, paragraph.text[:80])
for t, table in enumerate(doc.tables):
    print("table", t, [cell.text[:20] for cell in table.rows[0].cells])
```

Text in tables, headers and footers isn't in `doc.paragraphs`. Reach it through `table.cell(r, c)`
and `doc.sections[0].header.paragraphs`.

## Replace text and keep its formatting

Word splits a phrase across runs wherever its formatting or edit history changes, so a phrase you
can see often isn't inside one run. Setting `paragraph.text` would drop every run's font, size and
color, and `run.text` would drop a picture that shares the run. This replaces the first match in a
paragraph, across runs, and changes only text: the new text takes the formatting of the run where
the old text started. Call it again for the next match.

```python
def replace(paragraph, old, new):
    texts = [t for run in paragraph.runs for t in run._r.findall(qn("w:t"))]
    joined = "".join(t.text or "" for t in texts)
    start = joined.find(old)
    if start < 0:
        return False
    end, position = start + len(old), 0
    for t in texts:
        text = t.text or ""
        t_start, t_end = position, position + len(text)
        position = t_end
        if t_end <= start or t_start >= end:
            continue
        t.text = (text[:max(0, start - t_start)] + (new if t_start <= start < t_end else "")
                  + (text[end - t_start:] if t_end > end else ""))
        t.set(qn("xml:space"), "preserve")  # keeps spaces at either end
    return True
```

## Insert, move and delete

- New content goes at the end. To place it, add it and then move its element:
  `paragraph._p.addnext(new_paragraph._p)` or `paragraph._p.addnext(table._tbl)`.
- Delete a paragraph: `paragraph._p.getparent().remove(paragraph._p)`.
- A new item in an existing list copies an item that's there, so it keeps the list's numbering and
  style. Style names such as "List Number" exist only in some files.

  ```python
  import copy

  def add_after(paragraph, text):
      new = copy.deepcopy(paragraph._p)
      paragraph._p.addnext(new)
      added = type(paragraph)(new, paragraph._parent)
      runs = added.runs or [added.add_run()]
      runs[0].text = text
      for run in runs[1:]:
          run.text = ""
      return added
  ```

## Comments

Comments suit feedback on a student's work, since the student's own text stays as it was:

```python
paragraph = doc.paragraphs[4]
doc.add_comment(paragraph.runs, text="Which evidence supports this claim?", author="Ms. Rivera", initials="MR")
```

`runs` may be one run or several in a row; the comment marks the text they cover. Comments don't
show in review_file's pages or in the PDF copy deliver_file sends, so check them with
`[(c.author, c.text) for c in Document(path).comments]`, and tell the user they're in the Word file.

## A shaded callout box

A one-cell table with a fill:

```python
def callout(doc, text, fill="E8EEF5"):
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    cell.text = text
    shade = OxmlElement("w:shd")
    shade.set(qn("w:val"), "clear")
    shade.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shade)
    return table
```

## Header, logo and columns

```python
section = doc.sections[0]
header = section.header.paragraphs[0]
header.text = "Ms. Rivera · Biology · Room 214"
header.runs[0].add_picture("logo.png", width=Inches(0.6))  # a picture goes inside a run

columns = section._sectPr.find(qn("w:cols"))  # Word's files have one already
if columns is None:
    columns = OxmlElement("w:cols")
    section._sectPr.append(columns)
columns.set(qn("w:num"), "2")  # two newspaper columns for this section
columns.set(qn("w:space"), "720")  # twentieths of a point: 720 is half an inch
```

## Pictures side by side

A borderless table with one picture per cell:

```python
table = doc.add_table(rows=1, cols=2)
for cell, path in zip(table.rows[0].cells, ["before.png", "after.png"]):
    cell.paragraphs[0].add_run().add_picture(path, width=Inches(3))
```

## What python-docx can't do

- Tracked changes. Say so, and make the edits directly, or suggest them as comments.
- Update a table of contents or other fields. Word fills them in when the user opens the file.
