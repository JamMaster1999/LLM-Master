---
name: pdf
description: Makes designed PDFs from HTML and CSS with WeasyPrint, and merges, splits, rotates, stamps and fills existing PDFs with pypdf. Use this skill when the user asks for a PDF by name, or for a print layout a word processor handles badly, such as flashcards, exit tickets to cut apart, a certificate, a poster or a sign, and when an uploaded PDF needs combining, splitting, filling in or its tables pulled out. Worksheets and handouts go to docx, which sends a PDF copy too.
---

# PDFs

## New PDFs

Write the page as HTML, link the shared stylesheet, and render it:

```bash
weasyprint --pdf-tags handout.html handout.pdf
```

`--pdf-tags` marks headings, lists and tables, so screen readers can follow the PDF. It needs the
page's language, as in `<html lang="en">`.

`references/example.html` is a two-page handout with a name line, a shaded box, a table, writing
lines and cut-apart exit tickets. Start from it. `assets/base.css` gives Letter pages with page
numbers, Calibri, ruled tables, and these classes:

| class | gives |
|---|---|
| `subtitle` | a gray line under the `h1` |
| `name-line` | `Name: ____ Date: ____`, with a `<span>` for each blank |
| `line` | one ruled line to write on; repeat the div for more |
| `box` | a shaded box that stays on one page |
| `grid`, `grid three` | two or three columns |
| `card` | a dashed, fixed-height card to cut out |
| `page-break` | starts a new page |
| `small` | small gray text for sources and notes |

Change the colors in the page's own `<style>`: `:root { --accent: #7A3E9D; --tint: #F1E8F7; }`.
For A4 or landscape, add `@page { size: A4; }` or `@page { size: Letter landscape; }`.

## What WeasyPrint can't reach

The sandbox has no network, and WeasyPrint runs no JavaScript.

- **Images** must be files in the sandbox, such as those `generate_image` saves under `images/`.
  A remote URL renders as nothing.
- **Fonts** must be installed: Calibri, Cambria, Arial, Times New Roman, DejaVu and Noto. A web
  font link prints an error, yet the PDF still renders, in a fallback font.
- **Math** has no renderer here. For fractions and equations, use the docx skill: its document goes
  out with a PDF copy.
- **Emoji** have no font. Use words or an image.

## Other files to PDF

deliver_file sends a document or deck together with its PDF copy. Convert one yourself only when the
user wants the PDF alone: `soffice --headless --convert-to pdf --outdir . worksheet.docx`.

## Changing a PDF

Read `references/pypdf.md`. It covers merging, picking pages, rotating, filling form fields,
stamping every page, and pulling tables out. To read a PDF's text, use `read`.

## Check before delivery

1. Call `review_file` on the PDF and look at every page.
2. Read any page that looks off with `read(path, pages="2")`. Look for a box or table split
   across pages, text running into the margin, an empty last page, and missing images.
3. Fix the HTML, render again, and review again. Then deliver it.

## Sources

Put each source's number after the claim it supports, and end with a sources section listing each
number, title and URL.
