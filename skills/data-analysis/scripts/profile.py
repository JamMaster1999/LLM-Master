"""A first look at a data file: its size, and per column the type, missing values, distinct values,
and the range or most common values. Reads CSV and TSV, Excel and OpenDocument sheets (every
sheet), and JSON.

    python profile.py scores.csv [--encoding cp1250]

A text file is read as UTF-16 when it starts with that mark, as UTF-8 when it decodes as UTF-8, and
otherwise as Windows-1252, which Excel on Windows writes; --encoding names another. Lines above the
header, such as a title, are skipped. It prints how it read the file, for a script to do the same."""
import argparse
import codecs
import csv
import io
import json
from pathlib import Path

import pandas as pd


def load(path: str, encoding: str | None) -> dict[str, pd.DataFrame]:
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls", ".ods"):
        return pd.read_excel(path, sheet_name=None)
    if suffix == ".json":  # lists and objects inside records become text, so they can be counted
        frame = pd.read_json(path)
        return {"": frame.map(lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value)}
    data = Path(path).read_bytes()
    if encoding is None:
        encoding = "utf-16" if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"
        try:
            data.decode(encoding)
        except UnicodeDecodeError:
            encoding = "cp1252"
    text = data.decode(encoding, errors="replace")
    sample = text[:20_000]
    try:
        separator = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:  # one column, or nothing to go on
        separator = ","
    rows = list(csv.reader(io.StringIO(sample), delimiter=separator))[:50]
    widths = [len(row) for row in rows if row]
    width = max(set(widths), key=widths.count) if widths else 0
    skip = next((i for i, row in enumerate(rows) if len(row) == width), 0)  # the header is the first full row
    print(f"Read as {encoding}, columns split on {separator!r}" + (f", skipping {skip} lines above the header"
                                                                    if skip else "") + ".")
    return {"": pd.read_csv(io.StringIO(text), sep=separator, skiprows=skip)}


def describe(column: pd.Series) -> str:
    present = column.dropna()
    if present.empty:
        return "all missing"
    if pd.api.types.is_bool_dtype(column):
        return f"true {int(present.sum())}, false {int((~present.astype(bool)).sum())}"
    if pd.api.types.is_numeric_dtype(column):
        return (f"min {present.min():g}, median {present.median():g}, mean {present.mean():.4g}, "
                f"max {present.max():g}")
    if pd.api.types.is_datetime64_any_dtype(column):
        return f"from {present.min()} to {present.max()}"
    top = present.astype(str).value_counts().head(4)
    shown = ", ".join(f"{value[:30]!r} ×{count}" for value, count in top.items())
    return f"top: {shown}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--encoding", help="the text file's encoding, when the default reads it wrong")
    args = parser.parse_args()
    for i, (sheet, frame) in enumerate(load(args.path, args.encoding).items()):
        print(f"{chr(10) if i else ''}{'Sheet ' + repr(sheet) + ': ' if sheet else ''}{len(frame)} rows × "
              f"{len(frame.columns)} columns")
        unnamed = [name for name in frame.columns if str(name).startswith("Unnamed")]
        if unnamed:
            print(f"  {len(unnamed)} unnamed columns: the header may not be on the first row")
        duplicates = int(frame.duplicated().sum())
        if duplicates:
            print(f"  {duplicates} duplicate rows")
        empty = [name for name in frame.columns if frame[name].isna().all()]
        if empty and sheet:
            print(f"  {len(empty)} columns are empty. Formulas never computed read as empty: run the xlsx "
                  "skill's recalc.py on a copy first")
        for name in frame.columns:
            column = frame[name]
            missing = int(column.isna().sum())
            print(f"  {str(name)[:40]!r}: {column.dtype}, {missing} missing, {column.nunique()} distinct; "
                  f"{describe(column)}")


if __name__ == "__main__":
    main()
