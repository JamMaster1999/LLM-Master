"""read's work, done inside the sandbox where Pillow, openpyxl, LibreOffice and poppler live: one
call in, one JSON result out, so a read is one round trip and only its result leaves the sandbox.
The host runs this file's source with `python -c` and imports it for pi's text limits."""
import base64
import codecs
import csv
import glob
import io
import json
import os
import subprocess
import sys
import tempfile

MAX_LINES, MAX_BYTES = 2000, 50 * 1024  # pi's cut for long text
MAX_PAGES = 10
SHEET_BYTES = 25 * 1024  # a spreadsheet up to this much CSV comes back whole
MAX_FILE = 50 * 1024 * 1024  # the most any tool pulls out of the sandbox at once
IMAGE_SIDE = 1568  # px on the long side
IMAGE_BYTES = 3_750_000  # an image passed through as is: 5 MB as base64, the strictest provider's cap

IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
PAGED = {".pdf", ".docx", ".doc", ".odt", ".rtf", ".pptx", ".ppt", ".odp"}
SHEETS = {".xlsx", ".xlsm", ".xls", ".ods", ".csv", ".tsv"}
MEDIA = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".flac", ".opus", ".mp4", ".m4v", ".mov", ".webm",
         ".mkv", ".avi"}


class Problem(Exception):
    """Something about the file the model can fix; its message goes back as the result."""


def read(path: str, shown: str, offset: int | None, limit: int | None) -> list[dict]:
    """The result's parts for the file at `path`, which the model calls `shown`."""
    extension = os.path.splitext(path)[1].lower()
    if extension in MEDIA:
        raise Problem(f"read can't return audio or video. Use bash (ffprobe, ffmpeg) to work with {shown}.")
    if not os.path.exists(path):
        raise Problem(f"File not found: {shown}")
    if os.path.isdir(path):
        raise Problem(f"{shown} is a directory. Use bash to list it.")
    size = os.path.getsize(path)
    if size > MAX_FILE:
        raise Problem(f"{shown} is {format_size(size)}, over the {format_size(MAX_FILE)} limit. Use bash to work with it.")
    with tempfile.TemporaryDirectory() as work:
        if extension in IMAGES:
            return _image(path)
        if extension in PAGED:
            return _pages(path, shown, offset or 1, min(limit or MAX_PAGES, MAX_PAGES), work)
        if extension in SHEETS:
            return [_text(_sheets(path, work))]
        with open(path, "rb") as f:
            data = f.read()
        if not _utf8(data[:8192]):
            raise Problem(f"{shown} is a binary file or not UTF-8 text. Use bash to inspect it.")
        return [_text(page(data.decode(errors="replace"), shown, offset, limit))]


def page(text: str, path: str, offset: int | None, limit: int | None) -> str:
    """pi's read window: from `offset`, at most `limit` lines, cut at 2000 lines or 50 KB, with a
    notice that says how to read on."""
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    total = len(lines)
    start = (offset or 1) - 1
    if offset and start >= total:
        raise Problem(f"Offset {offset} is beyond end of file ({total} lines total)")
    end = min(total, start + limit) if limit else total
    shown, size = [], 0
    for line in lines[start:end]:
        size += len(line.encode()) + 1
        if len(shown) == MAX_LINES or size > MAX_BYTES:
            break
        shown.append(line)
    if not shown and start < end:
        line = start + 1
        return (f"[Line {line} is {format_size(len(lines[start].encode()))}, exceeds {format_size(MAX_BYTES)} limit. "
                f"Use bash: sed -n '{line}p' {path} | head -c {MAX_BYTES}]")
    last = start + len(shown)
    out = "\n".join(shown)
    if last < end:
        cap = "" if len(shown) == MAX_LINES else f" ({format_size(MAX_BYTES)} limit)"
        out += f"\n\n[Showing lines {start + 1}-{last} of {total}{cap}. Use offset={last + 1} to continue.]"
    elif end < total:
        out += f"\n\n[{total - end} more lines in file. Use offset={end + 1} to continue.]"
    return out


def format_size(n: int) -> str:
    """Sizes the way pi writes them: 512B, 50.0KB, 1.2MB."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / 1024 / 1024:.1f}MB"


def _image(path: str) -> list[dict]:
    from PIL import Image, ImageOps  # pyright: ignore[reportMissingImports]  (the sandbox has these)
    import pillow_heif  # pyright: ignore[reportMissingImports]
    pillow_heif.register_heif_opener()
    image = Image.open(path)
    upright = image.getexif().get(0x0112, 1) == 1  # models read no EXIF, so a turned photo must be turned
    if (image.format in ("PNG", "JPEG", "GIF", "WEBP") and max(image.size) <= IMAGE_SIDE and upright
            and os.path.getsize(path) <= IMAGE_BYTES):
        mime = Image.MIME[image.format]
        with open(path, "rb") as f:
            data = f.read()
    else:  # too big, turned, or a format providers don't take: redraw it, JPEG unless it's transparent
        image = ImageOps.exif_transpose(image)
        image.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
        out, mime = io.BytesIO(), "image/png" if alpha else "image/jpeg"
        image.convert("RGBA" if alpha else "RGB").save(out, "PNG" if alpha else "JPEG", quality=85)
        data = out.getvalue()
    return [_text(f"Read image file [{mime}]"), _image_part(data, mime)]


def _pages(path: str, shown: str, first: int, count: int, work: str) -> list[dict]:
    if first < 1 or count < 1:
        raise Problem("offset and limit start at 1.")
    pdf = path if path.lower().endswith(".pdf") else _convert(path, "pdf", work)
    info = _run("pdfinfo", pdf)
    total = int(next(line.split()[1] for line in info.splitlines() if line.startswith("Pages:")))
    if first > total:
        raise Problem(f"Offset {first} is beyond the last page of {shown} ({total} pages)")
    last = min(first + count - 1, total)
    _run("pdftoppm", "-jpeg", "-jpegopt", "quality=85", "-scale-to", str(IMAGE_SIDE), "-f", str(first),
         "-l", str(last), pdf, f"{work}/page")
    header = f"Rendered {shown} [pages {first}-{last} of {total}]"
    if last < total:
        header += f" Use offset={last + 1} to continue."
    images = []
    for page_file in sorted(glob.glob(f"{work}/page*.jpg")):
        with open(page_file, "rb") as f:
            images.append(_image_part(f.read(), "image/jpeg"))
    return [_text(header), *images]


def _sheets(path: str, work: str) -> str:
    """Every sheet as CSV when it all fits in 25 KB, else a preview of each."""
    name = os.path.basename(path)
    if path.lower().endswith((".csv", ".tsv")):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            sheets = {name: list(csv.reader(f, delimiter="\t" if path.lower().endswith(".tsv") else ","))}
    else:
        import warnings
        import openpyxl  # pyright: ignore[reportMissingModuleSource]
        warnings.simplefilter("ignore")  # openpyxl warns about every extension it skips
        book = path if path.lower().endswith((".xlsx", ".xlsm")) else _convert(path, "xlsx", work)
        values, formulas = openpyxl.load_workbook(book, data_only=True), openpyxl.load_workbook(book)
        # a workbook built by code has formulas but no saved values: show the formula there
        sheets = {sheet.title: [[value if value is not None else formula for value, formula in zip(row, formula_row)]
                                for row, formula_row in zip(sheet.iter_rows(values_only=True),
                                                            formulas[sheet.title].iter_rows(values_only=True))]
                  for sheet in values.worksheets}
    whole = "\n".join(f"## {title}\n{_csv(rows)}" for title, rows in sheets.items()).rstrip("\n")
    if len(whole.encode()) <= SHEET_BYTES:
        return whole
    previews = [f"## {title}\n{_csv(rows[:20])[:SHEET_BYTES]}[{len(rows)} rows x {max(map(len, rows), default=0)} "
                f"columns; showing the first 20]" for title, rows in sheets.items()]
    return "\n\n".join(previews) + (f"\n\n[The full data is {len(whole.encode()) // 1024} KB as CSV, over the "
                                    f"{SHEET_BYTES // 1024} KB limit. It stays at {path}; use bash or run_program "
                                    f"to analyze it.]")


def _csv(rows: list) -> str:
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(["" if cell is None else cell for cell in row] for row in rows)
    return text.getvalue()


def _convert(path: str, to: str, work: str) -> str:
    """LibreOffice's copy of the file in another format, with a profile of its own so parallel
    conversions don't fight over one."""
    _run("soffice", f"-env:UserInstallation=file://{work}/profile", "--headless", "--convert-to", to,
         "--outdir", work, path)
    converted = glob.glob(f"{work}/*.{to}")
    if not converted:
        raise Problem(f"LibreOffice couldn't open {os.path.basename(path)}.")
    return converted[0]


def _run(*args: str) -> str:
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode:
        raise Problem(f"{args[0]} failed: {done.stderr.strip() or 'no reason given'}")
    return done.stdout


def _utf8(head: bytes) -> bool:
    """Whether the start of a file reads as UTF-8 text; a character cut at the end doesn't count."""
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head)
    except UnicodeDecodeError:
        return False
    return b"\0" not in head


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_part(data: bytes, mime: str) -> dict:
    return {"type": "image", "data": base64.b64encode(data).decode(), "mime": mime}


if __name__ == "__main__":
    request = json.loads(sys.argv[1])
    try:
        print(json.dumps({"content": read(**request)}))
    except Problem as problem:
        print(json.dumps({"problem": str(problem)}))
