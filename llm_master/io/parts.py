"""Message parts. Image, File and Audio read a data-uri, URL or path and hold base64 from creation on. See IO.md."""
import base64
import binascii
import io
import mimetypes
import pathlib
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
import pillow_heif
from PIL import Image as Pillow
from pydantic import BaseModel, Field

pillow_heif.register_heif_opener()
_AUDIO_FORMATS = {"audio/mpeg": "mp3", "audio/x-wav": "wav", "audio/wave": "wav", "audio/mp4a-latm": "m4a"}


class PartError(Exception):
    """A part's source couldn't be read or converted. Raised at creation, before any model is billed."""


class Text(BaseModel, extra="forbid"):
    type: Literal["text"] = "text"
    text: str


class Image(BaseModel, extra="forbid"):
    """Image("photo.heic"), Image("https://..."), Image("data:image/png;base64,...") — HEIC becomes PNG."""

    type: Literal["image"] = "image"
    data: str  # base64
    mime: str

    def __init__(self, source: str | None = None, /, **fields: Any):
        if source is not None:
            raw, mime = _read(source)
            if not mime.startswith("image/"):
                raise PartError(f"{source[:100]} is {mime}, not an image")
            if mime in ("image/heic", "image/heif"):
                raw, mime = _png(source, raw), "image/png"
            fields |= {"data": _b64(raw), "mime": mime}
        super().__init__(**fields)


class File(BaseModel, extra="forbid"):
    """File("report.pdf"): a PDF. The filename defaults to the source's last path segment."""

    type: Literal["file"] = "file"
    data: str  # base64
    mime: Literal["application/pdf"] = "application/pdf"
    filename: str

    def __init__(self, source: str | None = None, /, **fields: Any):
        if source is not None:
            raw, _ = _read(source)
            if not raw.startswith(b"%PDF"):
                raise PartError(f"{source[:100]} is not a PDF")
            name = "file.pdf" if source.startswith("data:") else pathlib.PurePosixPath(urlparse(source).path).name
            fields = {"data": _b64(raw), "filename": name or "file.pdf", **fields}
        super().__init__(**fields)


class Audio(BaseModel, extra="forbid"):
    """Audio("clip.wav"): user messages only. format is the short name (wav, mp3, ...)."""

    type: Literal["audio"] = "audio"
    data: str  # base64
    format: str

    def __init__(self, source: str | None = None, /, **fields: Any):
        if source is not None:
            raw, mime = _read(source)
            fields |= {"data": _b64(raw), "format": _AUDIO_FORMATS.get(mime, mime.removeprefix("audio/"))}
        super().__init__(**fields)


Part = Annotated[Text | Image | File | Audio, Field(discriminator="type")]


def uri(mime: str, data: str) -> str:
    return f"data:{mime};base64,{data}"


def _read(source: str) -> tuple[bytes, str]:
    try:
        if source.startswith("data:"):
            header, _, data = source.partition(",")
            return base64.b64decode(data, validate=True), header.removeprefix("data:").split(";")[0]
        if source.startswith(("http://", "https://")):
            reply = httpx.get(source, follow_redirects=True, timeout=30).raise_for_status()
            mime = reply.headers.get("content-type", "").split(";")[0] or mimetypes.guess_type(source)[0]
            return reply.content, mime or "application/octet-stream"
        return pathlib.Path(source).read_bytes(), mimetypes.guess_type(source)[0] or "application/octet-stream"
    except (OSError, httpx.HTTPError, binascii.Error) as e:
        raise PartError(f"can't read {source[:100]}: {e}") from e


def _png(source: str, raw: bytes) -> bytes:
    try:
        out = io.BytesIO()
        Pillow.open(io.BytesIO(raw)).save(out, "PNG")
        return out.getvalue()
    except (OSError, ValueError) as e:
        raise PartError(f"can't convert {source[:100]} to PNG: {e}") from e


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()
