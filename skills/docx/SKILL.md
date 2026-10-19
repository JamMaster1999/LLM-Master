---
name: docx
description: Builds Word documents (.docx) from Markdown through pandoc in a clean house style, and changes uploaded ones with python-docx. Use this skill whenever the user wants a Word document or a .docx file, such as a worksheet, handout, study guide, lesson plan, letter, syllabus or report, and when they upload a .docx to change.
---

# Word documents

## New documents

Write the content as Markdown, then convert it:

```bash
python /skills/docx/scripts/md2docx.py worksheet.md worksheet.docx
```

The script runs pandoc with a house style: Calibri, dark blue headings, ruled tables with a shaded
header row, page numbers in the footer, and 1-inch margins on Letter paper. Pass `--a4` for users
outside the US and Canada, `--landscape` for wide tables, and `--size 14` or more for young
readers; the body is 11 pt otherwise, and headings scale with it.

What the Markdown can hold:

- **A title block** at the top, which prints as the document's title and subtitle:

  ```markdown
  ---
  title: "Phases of the Moon"
  subtitle: "Grade 6 Science · Practice worksheet"
  ---
  ```

- **Headings, lists, bold and italic, links and footnotes** (`text[^1]`, then `[^1]: A note.`).
- **Pipe tables.** Keep cells short; a cell can't hold a list.
- **Math** in TeX, `$\frac{3}{4}$` inline or `$$...$$` on its own line. It becomes Word's own
  equations, which the user can edit.
- **Images,** files in the sandbox, with a width in inches: `![A labeled leaf](images/leaf.png){width=3in}`.
  The text is 6.5 inches wide on Letter, 6.3 on A4 and 9 in landscape. Percent widths are ignored.
- **`\newpage`** alone on a line starts a new page, such as before an answer key.
- **`\lines 3`** alone on a line draws three ruled lines to write on. A run of underscores alone on
  a line becomes a page-wide rule instead. Inline blanks such as `Name: __________` are fine.
- **Checkboxes** as lines that start with `| ☐`, one per line: `| ☐ I showed my work.` A list item
  would print a bullet beside the box.
- **No emoji.** The sandbox has no emoji font, so they print as boxes in the PDF copy.

For a layout Markdown can't express, such as a letterhead, two columns or a shaded callout box,
convert first and then finish the file with python-docx. `references/python-docx.md` has the
snippets.

## Check before delivery

1. Call `review_file` on the .docx and look at every page.
2. Read pages that look off with `read(path, pages="2")`. Look for tables split badly across pages,
   images too large or too small, a heading alone at the bottom of a page, and empty pages.
3. Fix the Markdown, convert again, and review again. Then deliver it.

## Documents for students

- Put the student's name and date line at the top of anything handed out.
- Leave room to answer: `\lines` under open questions, a blank table column for vocabulary.
- Put an answer key after `\newpage`, or in a separate file when students get the document
  digitally.
- Keep the body at 11 pt or more, and 14 pt or more for young readers. Don't shrink text to save a
  page.

## Sources

Put each source's number after the claim it supports, and end with a Sources section listing each
number, title and URL.

## Changing an uploaded document

Read `references/python-docx.md` first. Save under a new name, and keep the document's own styles.
For feedback on a student's work, add comments rather than rewriting their text. A legacy `.doc`
converts first: `soffice --headless --convert-to docx --outdir . uploads/old.doc`.
