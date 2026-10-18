"""Message parts. Image, File and Audio read a data-uri, URL or path and hold base64 from creation on. See IO.md."""
import base64
import io
import pathlib
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
import pillow_heif
from PIL import Image as Pillow
from pydantic import BaseModel, Field

pillow_heif.register_heif_opener()
_AUDIO_ALIASES = {"mpeg": "mp3", "wave": "wav"}  # declared subtypes whose short name differs


class PartError(Exception):
    """A part's source couldn't be read or converted. Raised at creation, before any model is billed."""


class Text(BaseModel, extra="forbid"):
    type: Literal["text"] = "text"
    text: str


class Image(BaseModel, extra="forbid"):
    """Image("photo.heic"), Image("https://..."), Image("data:image/png;base64,..."). HEIC becomes JPEG."""

    type: Literal["image"] = "image"
    data: str  # base64
    mime: str

    def __init__(self, source: str | None = None, /, **fields: Any):
        if source is not None:
            raw, _, _ = _read(source)
            try:
                with Pillow.open(io.BytesIO(raw)) as img:  # the bytes say what they are, not the name
                    # a phone's multi-picture JPEG is a JPEG to every provider
                    mime = "image/jpeg" if img.format == "MPO" else img.get_format_mimetype()
                    if img.format == "HEIF":  # a phone photo as PNG runs ~18 MB, past Claude's 10 MB cap
                        out = io.BytesIO()
                        img.convert("RGB").save(out, "JPEG", quality=90)
                        raw, mime = out.getvalue(), "image/jpeg"
            except (OSError, ValueError, Pillow.DecompressionBombError) as e:  # UnidentifiedImageError is an OSError
                raise PartError(f"{source[:100]} isn't a readable image: {e}") from e
            fields = {"data": _b64(raw), "mime": mime, **fields}
        super().__init__(**fields)


class File(BaseModel, extra="forbid"):
    """File("report.pdf"): a PDF. The filename defaults to the source's last path segment."""

    type: Literal["file"] = "file"
    data: str  # base64
    mime: Literal["application/pdf"] = "application/pdf"
    filename: str

    def __init__(self, source: str | None = None, /, **fields: Any):
        if source is not None:
            raw, _, name = _read(source)
            if not raw.startswith(b"%PDF"):
                raise PartError(f"{source[:100]} is not a PDF")
            fields = {"data": _b64(raw), "filename": name or "file.pdf", **fields}
        super().__init__(**fields)


class Audio(BaseModel, extra="forbid"):
    """Audio("clip.wav"): user messages only. format is the short name (wav, mp3, flac, ...)."""

    type: Literal["audio"] = "audio"
    data: str  # base64
    format: str

    def __init__(self, source: str | None = None, /, **fields: Any):
        if source is not None:
            raw, declared, name = _read(source)
            subtype = declared.split("/")[-1].removeprefix("x-")
            short = pathlib.PurePosixPath(name).suffix[1:].lower() or _AUDIO_ALIASES.get(subtype, subtype)
            fields = {"data": _b64(raw), "format": short, **fields}
        super().__init__(**fields)


Part = Annotated[Text | Image | File | Audio, Field(discriminator="type")]


def uri(mime: str, data: str) -> str:
    return f"data:{mime};base64,{data}"


def decode(data_uri: str) -> tuple[bytes, str]:
    """A data-uri's bytes and its mime type."""
    header, _, data = data_uri.partition(",")
    return base64.b64decode(data, validate=True), header.removeprefix("data:").split(";")[0]


def _read(source: str) -> tuple[bytes, str, str]:
    """The bytes, the type the source declares ("" for a path), and its last path segment ("" for a data-uri)."""
    try:
        if source.startswith("data:"):
            return *decode(source), ""
        if source.startswith(("http://", "https://")):
            reply = httpx.get(source, follow_redirects=True, timeout=30).raise_for_status()
            declared = reply.headers.get("content-type", "").split(";")[0]
            return reply.content, declared, pathlib.PurePosixPath(urlparse(source).path).name
        return pathlib.Path(source).read_bytes(), "", pathlib.Path(source).name
    except (OSError, ValueError, httpx.HTTPError, httpx.InvalidURL) as e:  # binascii.Error is a ValueError
        raise PartError(f"can't read {source[:100]}: {e}") from e


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()
