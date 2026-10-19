# Changing PDFs with pypdf, and pulling tables out with pdfplumber

Write the result to a new file; never overwrite the upload.

## Merge, pick pages, rotate

```python
from pypdf import PdfReader, PdfWriter

writer = PdfWriter()
for path in ["uploads/part1.pdf", "uploads/part2.pdf"]:
    writer.append(path)                      # whole files, in order
writer.append("uploads/notes.pdf", pages=(0, 3))  # pages 1-3 only (0-based, end excluded)
writer.pages[0].rotate(90)                   # clockwise, in multiples of 90
writer.write("combined.pdf")
```

## Fill a form

```python
reader = PdfReader("uploads/permission-slip.pdf")
for name, field in (reader.get_fields() or {}).items():
    print(name, field.get("/V"), field.get("/_States_", ""))  # a checkbox lists its states, such as /on

writer = PdfWriter(clone_from=reader)
writer.update_page_form_field_values(None, {"Student Name": "Ana Lopez", "Grade": "4", "Agree": "/on"},
                                     auto_regenerate=False)  # None: fields on every page
writer.write("permission-slip-filled.pdf")
```

A checkbox takes the state it lists other than `/Off`. `auto_regenerate=False` keeps each field's own
drawing, so viewers show the filled values without redrawing them.

A PDF with no fields is a scan or a flat form: say so, and offer a fillable copy instead. Build it
in HTML with `<input name="Grade">` fields and render it with `weasyprint --pdf-forms form.html
form.pdf`.

## Stamp every page

Make the stamp as a one-page PDF with WeasyPrint, such as a "DRAFT" watermark or a header, then:

```python
stamp = PdfReader("stamp.pdf").pages[0]
writer = PdfWriter(clone_from="uploads/handout.pdf")
for page in writer.pages:
    page.merge_page(stamp)
writer.write("handout-stamped.pdf")
```

## Tables out of a PDF

`read` gives a PDF's text. For tables as rows, use pdfplumber:

```python
import pdfplumber

with pdfplumber.open("uploads/scores.pdf") as pdf:
    for number, page in enumerate(pdf.pages, 1):
        for table in page.extract_tables():
            print(number, table[:3])  # a list of rows, each a list of cell strings
```

A scanned PDF has no text layer, so pdfplumber finds nothing. `read` OCRs it instead.
