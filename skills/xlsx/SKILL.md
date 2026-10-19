---
name: xlsx
description: Builds Excel workbooks (.xlsx) with openpyxl, with live formulas that are computed and checked before delivery, and changes uploaded ones. Use this skill whenever the user wants a spreadsheet or an .xlsx file as the result, such as a gradebook, roster, sign-up sheet, tracker, budget or data table, and when they upload a spreadsheet to change. For analysis whose result is a chart or a summary, use data-analysis.
---

# Excel workbooks

## Workflow

1. **Build with openpyxl** in a script, such as `build_gradebook.py`.
2. **Use formulas for anything derived**: totals, averages, percentages, letter grades. The user
   edits a score and the sheet updates. Compute in Python only what the sheet shouldn't recompute.
3. **Recalculate:** `python /skills/xlsx/scripts/recalc.py gradebook.xlsx`. openpyxl saves formulas
   without values, so previews and pandas see empty cells, and `read` shows the formulas, until this
   runs. It saves the values into the file and lists every formula that gives an error. Fix them and
   run it again.
4. **Read it back** with `read`. Check that the numbers are right, not only error-free: an
   off-by-one range gives a clean, wrong total. Compare two or three with values you compute in
   Python.
5. **Deliver it.** Spreadsheets need no review_file.

## Formulas that work everywhere

- Excel 2007 functions work as written: `SUM`, `AVERAGE`, `IF`, `COUNTIF`, `SUMIFS`, `SUMPRODUCT`,
  `INDEX`, `MATCH`, `VLOOKUP`, `ROUND`, `IFERROR`.
- Newer functions need the `_xlfn.` prefix in openpyxl, or they come out as `#NAME?`:
  `_xlfn.XLOOKUP`, `_xlfn.TEXTJOIN`, `_xlfn.IFS`, `_xlfn.MAXIFS`, `_xlfn.MINIFS`, `_xlfn.CONCAT`.
- Avoid `FILTER`, `SORT`, `UNIQUE` and `SEQUENCE`. Their results spill over several cells, which an
  openpyxl file can't hold. Sort and filter in Python instead.
- Quote a sheet name that has a space: `='Unit 3'!B2`.
- Merged cells take a value only in their top-left cell.

## Gradebooks

- **Blank isn't zero.** `AVERAGE` skips blank cells and counts zeros. Leave missing work blank or
  enter 0 as the teacher's policy says, and say which you did.
- **Weights in labeled cells**, used through `SUMPRODUCT`, so the teacher can change a category's
  weight in one place.
- **Letter grades from a lookup table** on their own sheet, with `VLOOKUP(score, table, 2, TRUE)` on
  the table sorted by its lowest score. Nested `IF`s are hard to change.
- **Percentages stored as fractions:** 0.85 with the format `0%` shows 85%.

## Formatting

- A bold header row with a fill, frozen below it: `ws.freeze_panes = "A2"`.
- Column widths that fit the content: `ws.column_dimensions["A"].width = 24`.
- Number formats on every number column: `0.0`, `0%`, dates as real dates with `mm/dd/yyyy`.
- Conditional formatting for thresholds the teacher cares about, such as scores under 70.
- Dropdowns for columns people fill in: `DataValidation(type="list", formula1='"Yes,No"')`.
- In a template for others to fill in, shade the input cells pale yellow and add a short "How to
  use" note at the top.
- Native charts with `openpyxl.chart` when a chart helps; they recalculate with the data.

## Changing an uploaded workbook

- `load_workbook(path)` keeps formulas and formatting. Save under a new name.
- Never save a workbook opened with `data_only=True`: it holds values only, so every formula is lost.
- Don't write it back with pandas `to_excel`, which drops the formatting. Read data with pandas if
  that's easier, and write cells with openpyxl.
- An `.xlsm` loses its macros unless loaded with `keep_vba=True`, and recalc.py takes `.xlsx` only.
- An `.xls` or `.ods` converts to `.xlsx` first: `soffice --headless --convert-to xlsx --outdir .
  uploads/grades.xls`.
- Match the workbook's existing conventions: its fonts, colors and where inputs go.

## Sources

When numbers come from the web, add a Sources sheet, or a note under the table, listing each
source's number, title and URL.
