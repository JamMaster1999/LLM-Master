"""The file tools: read, write and edit copy pi's; review_file checks what the model made, and
deliver_file hands it to the user. And uploads, which the caller attaches to the user's message."""
import json
import mimetypes
import posixpath
import uuid
from importlib import resources
from typing import Annotated

from modal.exception import SandboxFilesystemError, SandboxFilesystemNotFoundError
from modal.types import FileType
from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict  # pydantic takes typing's only from Python 3.12

from agent.reader import DURATION, MAX_FILE, TRANSCRIPT, VIEWS, attached, family_of, format_size, view_hash
from agent.result import Result, ToolError, UserFile
from agent.sandbox import Sandbox, resolve
from agent.shell import MAX_ARGUMENT
from agent.transcribe import transcribe

READ_TIMEOUT = 300  # a view's first build: LibreOffice and Indra on up to 100 pages
_READER = resources.files("agent").joinpath("reader.py").read_text()
assert len(_READER.encode()) <= MAX_ARGUMENT, "reader.py goes to the sandbox as one exec argument"

READ = {"type": "function", "function": {
    "name": "read",
    "description": "Read the contents of a file. Supports text files, images (jpg, png, gif, webp, heic, avif, bmp, "
                   "tiff), PDFs, documents and slides (docx, pptx and others) and audio (mp3, m4a, wav and others) as "
                   "their transcript, ebooks and notebooks (epub, ipynb), and spreadsheets (xlsx, xls, ods, numbers, "
                   "as CSV). Images are sent as attachments. Output is truncated to 2000 lines or 50KB (whichever is "
                   "hit first). Use offset/limit for large files. When you need the full file, continue with offset "
                   "until complete. To see pages of a PDF, document or slides, use pages.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        "pages": {"type": "string", "description": "Page range for PDFs, documents and slides (e.g., \"3\" or "
                  "\"2-5\"), returned as page images. Maximum 10 pages per request."},
    }, "required": ["path"]}}}

WRITE = {"type": "function", "function": {
    "name": "write",
    "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
                   "Automatically creates parent directories.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    }, "required": ["path", "content"]}}}

EDIT = {"type": "function", "function": {
    "name": "edit",
    "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
                   "non-overlapping region of the original file. If two changes affect the same block or nearby "
                   "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
                   "unchanged regions just to connect distant changes.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {"type": "array", "description": "One or more targeted replacements. Each edit is matched against "
                  "the original file, not incrementally. Do not include overlapping or nested edits. If two changes "
                  "touch the same block or nearby lines, merge them into one edit instead.",
                  "items": {"type": "object", "properties": {
                      "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must "
                                  "be unique in the original file and must not overlap with any other "
                                  "edits[].oldText in the same call."},
                      "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                  }, "required": ["oldText", "newText"]}},
    }, "required": ["path", "edits"]}}}

REVIEW = {"type": "function", "function": {
    "name": "review_file",
    "description": "Check a PDF, document, slides or image before sending it with deliver_file, which refuses one "
                   "not reviewed since it last changed. Returns the page count, every page as a numbered thumbnail "
                   "on one image, and each page's first line; for an image, the image. To see a page in detail, use "
                   "read with pages.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the file to review (relative or absolute)"},
    }, "required": ["path"]}}}

DELIVER = {"type": "function", "function": {
    "name": "deliver_file",
    "description": "Send a file from the sandbox to the user, who can open or download it. Use it for every file "
                   "the user asked for, once it is finished. A PDF, document, slides or image must be reviewed with "
                   "review_file first. A document or slides also goes out as a PDF.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the file to send (relative or absolute)"},
    }, "required": ["path"]}}}


class Edit(TypedDict):
    oldText: str
    newText: str


_EDITS: TypeAdapter[list[Edit] | Edit] = TypeAdapter(list[Edit] | Edit)


async def read(sandbox: Sandbox, path: str, offset: Annotated[int, Field(ge=1)] | None = None,
               limit: Annotated[int, Field(ge=1)] | None = None, pages: str | int | None = None) -> Result:
    content = await _reader(sandbox, "read", path, offset=offset, limit=limit,
                            pages=None if pages is None else str(pages))
    return Result(content[0]["text"] if len(content) == 1 and content[0]["type"] == "text" else content)


async def review_file(sandbox: Sandbox, path: str) -> Result:
    return Result(await _reader(sandbox, "review", path))


async def mark_reviewed(sandbox: Sandbox, path: str, data: bytes) -> None:
    """Marks a file the model has already seen as reviewed, as review_file does."""
    await sandbox.write(f"{VIEWS}/{view_hash(data)}/reviewed", (resolve(path) + "\n").encode())


async def upload(sandbox: Sandbox, name: str, data: bytes) -> str:
    """Saves an upload unchanged under uploads/; a taken name gets a number, as in `notes (2).pdf`.
    Returns its path. Run one at a time, so two uploads can't take one name."""
    name = posixpath.basename(name)
    if not name.strip("."):  # "", "." and ".." would name the folder itself
        name = "upload"
    stem, extension = posixpath.splitext(name)
    number = 1
    while True:
        path = f"uploads/{stem}{f' ({number})' if number > 1 else ''}{extension}"
        try:
            await sandbox.stat(path)
        except SandboxFilesystemNotFoundError:
            await sandbox.write(path, data)
            return path
        number += 1


async def attach(sandbox: Sandbox, path: str) -> list[dict]:
    """The user message's parts for an upload: its path, and what the model gets within read's
    limits. A view that can't be built is a note, not a failure."""
    try:
        return await _reader(sandbox, "attach", path)
    except ToolError as error:
        return [{"type": "text", "text": attached(path, str(error))}]


async def _reader(sandbox: Sandbox, do: str, path: str, **arguments) -> list[dict]:
    """reader.py's parts for the file. A recording without a view is transcribed here first."""
    request = json.dumps({"do": do, "path": resolve(path), "shown": path, **arguments})
    answer = await _run_reader(sandbox, request, path)
    if "transcribe" in answer:
        await _transcribe(sandbox, path)
        answer = await _run_reader(sandbox, request, path)
        if "transcribe" in answer:
            raise ToolError(f"{path} changed while it was being transcribed. Read it again.")
    if "problem" in answer:
        raise ToolError(answer["problem"])
    return answer["content"]


async def _run_reader(sandbox: Sandbox, request: str, path: str) -> dict:
    try:
        # -I keeps /workspace off the import path, so a json.py there can't replace the reader's
        code, stdout, stderr = await sandbox.exec("python", "-I", "-c", _READER, request, timeout=READ_TIMEOUT)
    except TimeoutError:
        raise ToolError(f"Reading {path} took over {READ_TIMEOUT} seconds.") from None
    if code:  # a file its library couldn't open
        reason = stderr.decode(errors="replace").strip().splitlines() or ["unknown error"]
        raise ToolError(f"Could not read {path}: {reason[-1]}")
    return json.loads(stdout)


async def _transcribe(sandbox: Sandbox, path: str) -> None:
    """A recording's view, its transcript and length, under the hash of the bytes sent to Together.
    Like reader.py's views, it's made in a temp folder renamed into place."""
    data = await load(sandbox, path)
    view = f"{VIEWS}/{view_hash(data)}"
    duration, lines = await transcribe(posixpath.basename(path), data)
    work = f"{view}.{uuid.uuid4().hex[:8]}"
    await sandbox.write(f"{work}/{TRANSCRIPT}",
                        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode())
    await sandbox.write(f"{work}/{DURATION}", str(duration).encode())
    # a parallel build that got there first wins
    code, _, stderr = await sandbox.exec("bash", "-c", 'mv -T "$1" "$2" || { [ -d "$2" ] && rm -r "$1"; }', "bash",
                                         work, view, timeout=30)
    if code:
        raise ToolError(f"Could not save the transcript of {path}: {stderr.decode(errors='replace').strip()}")


async def write(sandbox: Sandbox, path: str, content: str) -> str:
    try:
        await sandbox.write(path, content.encode())
    except SandboxFilesystemError as e:
        raise ToolError(f"Could not write {path}: {e}") from None
    return f"Successfully wrote to {path}"


async def edit(sandbox: Sandbox, path: str, edits: list[Edit] | Edit | str) -> str:
    if isinstance(edits, str):  # pi: some models send the array as a JSON string
        try:
            edits = _EDITS.validate_json(edits)
        except ValidationError:
            raise ToolError("Edit tool input is invalid. edits must be an array of {oldText, newText}.") from None
    edits = edits if isinstance(edits, list) else [edits]  # and some send one edit without the array
    try:
        text = (await load(sandbox, path)).decode()
    except UnicodeDecodeError:
        raise ToolError(f"{path} is not UTF-8 text. Use bash to change it.") from None
    await write(sandbox, path, apply_edits(text, edits, path))
    return f"Successfully replaced {len(edits)} block(s) in {path}."


async def deliver_file(sandbox: Sandbox, path: str) -> Result:
    """Claude Code's rule that Edit refuses a file not yet read: a PDF, document, slides or image goes
    out only once its current version is reviewed. A document or slides goes with its view's PDF."""
    data = await load(sandbox, path)
    name = posixpath.basename(path)
    sent = [UserFile(name, mimetypes.guess_type(name)[0] or "application/octet-stream", data)]
    if (family := family_of(path)) in ("pages", "image"):
        view = f"{VIEWS}/{view_hash(data)}"
        try:
            await sandbox.stat(f"{view}/reviewed")
        except SandboxFilesystemNotFoundError:
            # each mark names the paths reviewed at that version, so an older one names this path
            code, _, _ = await sandbox.exec("grep", "-rqxF", "--include=reviewed", "--", resolve(path), VIEWS, timeout=30)
            raise ToolError(f"{path} has changed since it was reviewed. Review it again before sending it." if code == 0
                            else f"{path} has not been reviewed yet. Review it with review_file before sending it.") from None
        if family == "pages" and not name.lower().endswith(".pdf"):
            sent.append(UserFile(f"{posixpath.splitext(name)[0]}.pdf", "application/pdf",
                                 await sandbox.read(f"{view}/source.pdf")))
    return Result(f"Sent {' and '.join(file.name for file in sent)} to the user.", files=sent)


def apply_edits(text: str, edits: list[Edit], path: str) -> str:
    """pi's edit: every oldText matches the original exactly once, no two overlap, and the file keeps
    its BOM and line endings."""
    if not edits:
        raise ToolError("Edit tool input is invalid. edits must contain at least one replacement.")
    one = len(edits) == 1
    bom = "﻿" if text.startswith("﻿") else ""
    original = text.removeprefix(bom)
    crlf = "\r\n" in original
    original = original.replace("\r\n", "\n")
    spans = []
    for i, edit in enumerate(edits):
        old, new = edit["oldText"].replace("\r\n", "\n"), edit["newText"].replace("\r\n", "\n")
        name = "the text" if one else f"edits[{i}]"
        if not old:
            raise ToolError(f"oldText must not be empty in {path}." if one else f"edits[{i}].oldText must not be empty in {path}.")
        count = _occurrences(original, old)
        if count == 0:
            raise ToolError(f"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
                            if one else f"Could not find {name} in {path}. The oldText must match exactly including all whitespace and newlines.")
        if count > 1:
            raise ToolError(f"Found {count} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
                            if one else f"Found {count} occurrences of {name} in {path}. Each oldText must be unique. Please provide more context to make it unique.")
        start = original.index(old)
        spans.append((start, start + len(old), new, i))
    spans.sort()
    for (_, end, _, a), (start, _, _, b) in zip(spans, spans[1:]):
        if start < end:
            raise ToolError(f"edits[{a}] and edits[{b}] overlap in {path}. Merge them into one edit or target disjoint regions.")
    changed = original
    for start, end, new, _ in reversed(spans):
        changed = changed[:start] + new + changed[end:]
    if changed == original:
        raise ToolError(f"No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
                        if one else f"No changes made to {path}. The replacements produced identical content.")
    return bom + (changed.replace("\n", "\r\n") if crlf else changed)


def _occurrences(text: str, old: str) -> int:
    """How often old appears, overlapping ones too: "ana" is in "banana" twice."""
    count, start = 0, text.find(old)
    while start != -1:
        count, start = count + 1, text.find(old, start + 1)
    return count


async def load(sandbox: Sandbox, path: str) -> bytes:
    """A regular file's bytes, up to MAX_FILE, or a ToolError that says what's wrong with the path."""
    try:
        info = await sandbox.stat(path)
    except SandboxFilesystemNotFoundError:
        raise ToolError(f"File not found: {path}") from None
    if info.type == FileType.DIRECTORY:
        raise ToolError(f"{path} is a directory. Use bash to list it.")
    if info.type != FileType.FILE:
        raise ToolError(f"{path} is not a regular file.")
    if info.size > MAX_FILE:
        raise ToolError(f"{path} is {format_size(info.size)}, over the {format_size(MAX_FILE)} limit. Use bash to work with it.")
    return await sandbox.read(path)
