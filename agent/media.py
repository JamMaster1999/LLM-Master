"""The generate pair on llm_master: generate_image calls image(), generate_audio calls speech(). Each
saves what it made in the sandbox and sends it to the user."""
import base64
import io
import json
import posixpath
import uuid
from typing import Annotated

from PIL import Image as Pillow, UnidentifiedImageError
from pydantic import Field

from agent import files
from agent.result import Result, ToolError, UserFile
from agent.sandbox import Sandbox
from llm_master.facade import LLM
from llm_master.io.parts import Image
from llm_master.media import Media
from llm_master.types import Model

# the models core's media tests run live. gemini-3.1-flash-image and gemini-3.8-flash-tts join as
# the fallbacks once llm_master falls back (SPEC.md step 7); until then a call takes one candidate
IMAGE_MODELS = [Model(provider="openai", model="gpt-image-2")]
SPEECH_MODELS = [Model(provider="openai", model="gpt-4o-mini-tts")]
MAX_SPEECH = 4096  # characters, OpenAI's limit
EDITABLE = ("PNG", "JPEG", "WEBP")  # what OpenAI's edit endpoint takes

GENERATE_IMAGE = {"type": "function", "function": {
    "name": "generate_image",
    "description": "Generate an image from a text description, or edit an image by passing its path. The new image is "
                   "saved under images/, sent to the user, and shown to you. An edit starts from that image and your "
                   "prompt alone, not from earlier prompts, so describe the change fully.",
    "parameters": {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "What the image should show, or for an edit, the change to make"},
        "image": {"type": "string", "description": "Path to the image to edit (relative or absolute). Omit it to make "
                  "a new image"},
    }, "required": ["prompt"]}}}

GENERATE_AUDIO = {"type": "function", "function": {
    "name": "generate_audio",
    "description": "Turn text into speech. The audio is saved under audio/ and sent to the user. At most 4096 "
                   "characters per call.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "description": "The text to speak, at most 4096 characters"},
    }, "required": ["text"]}}}


async def generate_image(sandbox: Sandbox, llm: LLM, prompt: str, image: str | None = None) -> Result:
    parents = [Image(_editable(await files.load(sandbox, image), image))] if image else []
    made = await llm.image(IMAGE_MODELS, prompt, parents)
    path, sent = await _save(sandbox, "images", made)
    record = {"parent": image, "prompt": prompt, "candidate": made.origin.model_dump()}
    await sandbox.write(f"{posixpath.splitext(path)[0]}.json", json.dumps(record, ensure_ascii=False).encode())
    shown = await files.read(sandbox, path)  # read's image rule: at most 1568 px, JPEG unless transparent
    assert isinstance(shown.content, list)  # an image reads as text and the image
    shown.content[0]["text"] = f"Saved {path} and sent it to the user."
    return Result(shown.content, files=[sent])


async def generate_audio(sandbox: Sandbox, llm: LLM, text: Annotated[str, Field(max_length=MAX_SPEECH)]) -> Result:
    made = await llm.speech(SPEECH_MODELS, text)
    path, sent = await _save(sandbox, "audio", made)
    return Result(f"Saved {path} and sent it to the user.", files=[sent])


def _editable(data: bytes, path: str) -> str:
    """The parent as a data URI the edit endpoint takes: PNG, JPEG and WebP as they are, any other
    image redrawn as PNG."""
    try:
        with Pillow.open(io.BytesIO(data)) as picture:
            if picture.format not in EDITABLE:
                redrawn = io.BytesIO()
                picture.convert("RGBA").save(redrawn, "PNG")
                data = redrawn.getvalue()
    except UnidentifiedImageError:
        raise ToolError(f"{path} is not an image.") from None
    return "data:;base64," + base64.b64encode(data).decode()


async def _save(sandbox: Sandbox, folder: str, made: Media) -> tuple[str, UserFile]:
    """Saves the file as <folder>/<id>.<ext>; returns its path and the file for the user."""
    name = f"{uuid.uuid4().hex[:8]}.{made.mime.split('/')[1]}"
    await sandbox.write(f"{folder}/{name}", made.data)
    return f"{folder}/{name}", UserFile(name, made.mime, made.data)
