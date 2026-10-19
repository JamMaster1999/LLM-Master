"""The file tools: read, write and edit copy pi's; deliver_file hands a file to the user."""
import json
import mimetypes
import posixpath
from importlib import resources
from typing import Annotated

from modal.exception import SandboxFilesystemError, SandboxFilesystemNotFoundError
from modal.types import FileType
from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict  # pydantic takes typing's only from Python 3.12

from agent.reader import MAX_FILE, format_size
from agent.result import Result, ToolError, UserFile
from agent.sandbox import Sandbox, resolve

READ_TIMEOUT = 120  # LibreOffice on a long deck takes a while
_READER = resources.files("agent").joinpath("reader.py").read_text()

READ = {"type": "function", "function": {
    "name": "read",
    "description": "Read the contents of a file. Supports text files, images (jpg, png, gif, webp, bmp, tiff, heic), "
                   "PDFs and office documents (docx, pptx and others, shown as page images), and spreadsheets "
                   "(xlsx, xls, ods, csv, tsv, as CSV). Images are sent as attachments. For text files, output is "
                   "truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When "
                   "you need the full file, continue with offset until complete. For PDFs and documents, offset and "
                   "limit count pages, at most 10 per call.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed); for PDFs and documents, the first page"},
        "limit": {"type": "integer", "description": "Maximum number of lines to read; for PDFs and documents, pages (at most 10)"},
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

DELIVER = {"type": "function", "function": {
    "name": "deliver_file",
    "description": "Send a file from the sandbox to the user, who can open or download it. Use it for every file "
                   "the user asked for, once it is finished.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the file to send (relative or absolute)"},
    }, "required": ["path"]}}}


class Edit(TypedDict):
    oldText: str
    newText: str


_EDITS: TypeAdapter[list[Edit] | Edit] = TypeAdapter(list[Edit] | Edit)


async def read(sandbox: Sandbox, path: str, offset: Annotated[int, Field(ge=1)] | None = None,
               limit: Annotated[int, Field(ge=1)] | None = None) -> Result:
    request = json.dumps({"path": resolve(path), "shown": path, "offset": offset, "limit": limit})
    try:
        # -I keeps /workspace off the import path, so a json.py there can't replace the reader's
        code, stdout, stderr = await sandbox.exec("python", "-I", "-c", _READER, request, timeout=READ_TIMEOUT)
    except TimeoutError:
        raise ToolError(f"Reading {path} took over {READ_TIMEOUT} seconds.") from None
    if code:  # a file its library couldn't open
        reason = stderr.decode(errors="replace").strip().splitlines() or ["unknown error"]
        raise ToolError(f"Could not read {path}: {reason[-1]}")
    answer = json.loads(stdout)
    if "problem" in answer:
        raise ToolError(answer["problem"])
    content = answer["content"]
    return Result(content[0]["text"] if len(content) == 1 and content[0]["type"] == "text" else content)


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
    data = await load(sandbox, path)
    name = posixpath.basename(path)
    return Result(f"Sent {name} to the user.",
                  files=[UserFile(name, mimetypes.guess_type(name)[0] or "application/octet-stream", data)])


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
