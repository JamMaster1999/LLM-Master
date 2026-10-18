"""image() and speech(): media calls, one case per provider. See SPEC.md Media."""
import base64
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_never, cast

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion
from any_llm.utils.exception_handler import convert_exception
from openai import AsyncOpenAI

from llm_master.config import UnsupportedField
from llm_master.io.messages import Origin
from llm_master.io.parts import Image, uri
from llm_master.io.response import Attempt, Usage, completion_usage
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM

_SPEECH_MIMES = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/opus", "aac": "audio/aac",
                 "flac": "audio/flac", "pcm": "audio/pcm"}


class MissingMedia(Exception):
    """The candidate answered without the image or audio. Typed, so fallback moves on."""


@dataclass
class Media:
    """A generated file: its bytes and type, who made it, and what it took."""

    data: bytes
    mime: str
    origin: Origin
    usage: Usage | None  # openai speech reports none, so its cost is None, never zero
    attempts: list[Attempt]
    latency: float
    cost: float | None = None  # SPEC.md step 4


async def image(llm: "LLM", candidates: Model | Sequence[Model], prompt: str, images: Sequence[Image] = ()) -> Media:
    """A new image, or an edit: the parent image(s) plus the instruction, one request, no history."""
    candidate = _first(candidates)
    started = time.monotonic()
    match candidate.provider:
        case "openai":
            data, mime, usage = await _openai_image(llm, candidate, prompt, images)
        case "gemini":
            data, mime, usage = await _gemini_image(llm, candidate, prompt, images)
        case "vertexai":
            data, mime, usage = await _gemini_image(llm, candidate, prompt, images)
        case "anthropic" | "fireworks":
            raise UnsupportedField(candidate.provider, "image", "this provider makes no images")
        case _:
            assert_never(candidate.provider)
    return _media(candidate, data, mime, usage, started)


async def speech(llm: "LLM", candidates: Model | Sequence[Model], text: str) -> Media:
    """The text spoken, one shot."""
    candidate = _first(candidates)
    started = time.monotonic()
    match candidate.provider:
        case "openai":
            data, mime, usage = await _openai_speech(llm, candidate, text)
        case "gemini":
            data, mime, usage = await _gemini_speech(llm, candidate, text)
        case "vertexai":
            data, mime, usage = await _gemini_speech(llm, candidate, text)
        case "anthropic" | "fireworks":
            raise UnsupportedField(candidate.provider, "speech", "this provider makes no audio")
        case _:
            assert_never(candidate.provider)
    return _media(candidate, data, mime, usage, started)


def _first(candidates: Model | Sequence[Model]) -> Model:
    candidate = candidates if isinstance(candidates, Model) else candidates[0]  # fallback: SPEC.md step 7
    if extra := candidate.config.model_fields_set - {"timeout"}:
        raise UnsupportedField(candidate.provider, ", ".join(sorted(extra)), "media calls take only timeout")
    return candidate


def _options(candidate: Model) -> dict[str, Any]:
    """The candidate's extra, plus its timeout: on the held client, every key rides through to the provider."""
    timeout = candidate.config.timeout
    return {**({"timeout": timeout} if timeout is not None else {}), **candidate.extra}


async def _openai_image(llm: "LLM", candidate: Model, prompt: str, images: Sequence[Image]) -> tuple[bytes, str, Usage]:
    sdk = _openai(llm)
    try:
        if images:
            files = [(f"image{i}.{p.mime.split('/')[-1]}", base64.b64decode(p.data), p.mime) for i, p in enumerate(images)]
            reply = await sdk.images.edit(model=candidate.model, image=files, prompt=prompt, **_options(candidate))
        else:
            reply = await sdk.images.generate(model=candidate.model, prompt=prompt, **_options(candidate))
    except Exception as e:
        raise convert_exception(e, "openai") from e
    if not reply.data or not reply.data[0].b64_json:
        raise MissingMedia(f"openai/{candidate.model} answered without an image")
    u = reply.usage
    usage = Usage(input=u.input_tokens, output=u.output_tokens) if u else Usage()
    return base64.b64decode(reply.data[0].b64_json), f"image/{reply.output_format or 'png'}", usage


async def _gemini_image(llm: "LLM", candidate: Model, prompt: str, images: Sequence[Image]) -> tuple[bytes, str, Usage]:
    parents = [{"type": "image_url", "image_url": {"url": uri(p.mime, p.data)}} for p in images]
    reply = await _gemini(llm, candidate, [{"type": "text", "text": prompt}, *parents], "IMAGE")
    choice = reply.choices[0]
    if not choice.message.images:
        raise MissingMedia(f"{candidate.provider}/{candidate.model} answered without an image ({choice.finish_reason})")
    header, _, data = choice.message.images[0].image_url.url.partition(",")
    return base64.b64decode(data), header.removeprefix("data:").split(";")[0], completion_usage(reply.usage)


async def _openai_speech(llm: "LLM", candidate: Model, text: str) -> tuple[bytes, str, None]:
    options = _options(candidate)
    voice = options.pop("voice", "alloy")  # the API requires one
    try:
        reply = await _openai(llm).audio.speech.create(
            model=candidate.model, input=text, voice=voice, **options)
    except Exception as e:
        raise convert_exception(e, "openai") from e
    return reply.content, _SPEECH_MIMES.get(options.get("response_format", "mp3"), "audio/mpeg"), None


async def _gemini_speech(llm: "LLM", candidate: Model, text: str) -> tuple[bytes, str, Usage]:
    reply = await _gemini(llm, candidate, text, "AUDIO")
    audio = reply.choices[0].message.audio
    if not audio or not audio.data:
        raise MissingMedia(f"{candidate.provider}/{candidate.model} answered without audio")
    return base64.b64decode(audio.data), "audio/wav", completion_usage(reply.usage)  # any-llm wraps gemini's PCM in WAV


async def _gemini(llm: "LLM", candidate: Model, content: str | list[dict], modality: str) -> ChatCompletion:
    reply = await llm.client(candidate.provider).acompletion(
        model=candidate.model, messages=[{"role": "user", "content": content}], response_modalities=[modality],
        **_options(candidate))
    return cast(ChatCompletion, reply)  # not streamed, no response_format


def _openai(llm: "LLM") -> AsyncOpenAI:
    """The held openai SDK client. any-llm skips image edits and forbids extra keys on generate and speech."""
    return cast(BaseOpenAIProvider, llm.client("openai")).client


def _media(candidate: Model, data: bytes, mime: str, usage: Usage | None, started: float) -> Media:
    latency = time.monotonic() - started
    return Media(data=data, mime=mime, origin=Origin(provider=candidate.provider, model=candidate.model), usage=usage,
                 attempts=[Attempt(candidate.provider, candidate.model, latency)], latency=latency)
