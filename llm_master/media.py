"""image() and speech(): media calls, one case per provider, on query()'s pipeline. See SPEC.md Media."""
import base64
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, assert_never, cast

from any_llm.exceptions import ProviderError
from any_llm.types.completion import ChatCompletion
from openai._streaming import SSEDecoder

from llm_master import pipeline
from llm_master.config import UnsupportedField, merge
from llm_master.io.messages import Origin
from llm_master.io.parts import Image, decode, uri
from llm_master.io.response import Attempt, Usage, completion_usage
from llm_master.types import Model

if TYPE_CHECKING:
    from llm_master.facade import LLM

class MissingMedia(Exception):
    """The candidate answered without the image or audio. Typed, so fallback moves on."""


@dataclass
class Media:
    """A generated file: its bytes and type, who made it, and what it took."""

    data: bytes
    mime: str
    origin: Origin
    usage: Usage | None  # None when the provider reports none: cost None, never zero
    attempts: list[Attempt] = field(default_factory=list)  # the pipeline sets these two
    latency: float = 0.0
    cost: float | None = None  # SPEC.md step 4


async def image(llm: "LLM", candidates: Model | Sequence[Model], prompt: str, images: Sequence[Image] = ()) -> Media:
    """A new image, or an edit: the parent image(s) plus the instruction, one request, no history."""
    async def once(candidate: Model) -> Media:
        match candidate.provider:
            case "openai":
                return await _openai_image(llm, candidate, prompt, images)
            case "gemini":
                return await _gemini_image(llm, candidate, prompt, images)
            case "vertexai":
                return await _gemini_image(llm, candidate, prompt, images)
            case "anthropic":
                raise UnsupportedField("image", "provider", "anthropic makes no images")
            case "fireworks":
                raise UnsupportedField("image", "provider", "fireworks makes no images")
            case _:
                assert_never(candidate.provider)
    return await _make("image", candidates, once)


async def speech(llm: "LLM", candidates: Model | Sequence[Model], text: str) -> Media:
    """The text spoken, one shot."""
    async def once(candidate: Model) -> Media:
        match candidate.provider:
            case "openai":
                return await _openai_speech(llm, candidate, text)
            case "gemini":
                return await _gemini_speech(llm, candidate, text)
            case "vertexai":
                return await _gemini_speech(llm, candidate, text)
            case "anthropic":
                raise UnsupportedField("speech", "provider", "anthropic makes no audio")
            case "fireworks":
                raise UnsupportedField("speech", "provider", "fireworks makes no audio")
            case _:
                assert_never(candidate.provider)
    return await _make("speech", candidates, once)


async def _make(call: str, candidates: Model | Sequence[Model], once: Callable[[Model], Awaitable[Media]]) -> Media:
    chain = pipeline.chain(candidates)
    for candidate in chain:
        if set_fields := candidate.config.model_dump(exclude_defaults=True).keys() - {"timeout"}:
            raise UnsupportedField(call, min(set_fields), "media calls take only timeout")
        if candidate.extra.get("n", 1) != 1:  # a Media holds one file: more would be dropped unseen
            raise UnsupportedField(call, "n", "a media call makes one file; make one call per file")
    return await pipeline.run(chain, once)


def _made(candidate: Model, data: bytes, mime: str, usage: Usage | None) -> Media:
    return Media(data=data, mime=mime, origin=Origin(provider=candidate.provider, model=candidate.model), usage=usage)


def _request(candidate: Model, base: dict[str, Any]) -> dict[str, Any]:
    """Our keys, then the timeout, then the candidate's extra: the caller's last word wins, as in query()."""
    timeout = candidate.config.timeout
    return merge(base, {"timeout": timeout} if timeout is not None else {}, candidate.extra)


async def _openai_image(llm: "LLM", candidate: Model, prompt: str, images: Sequence[Image]) -> Media:
    files = [(f"image{i}.{p.mime.split('/')[-1]}", base64.b64decode(p.data), p.mime) for i, p in enumerate(images)]
    request = _request(candidate, {"model": candidate.model, "prompt": prompt} | ({"image": files} if files else {}))
    openai = llm.client("openai")
    reply = await (openai.aimage_edit(**request) if files else openai.aimage_generation(**request))
    if not reply.data or not reply.data[0].b64_json:
        raise MissingMedia(f"openai/{candidate.model} answered without an image")
    u = reply.usage
    usage = Usage(input=u.input_tokens, output=u.output_tokens) if u else None
    return _made(candidate, base64.b64decode(reply.data[0].b64_json), f"image/{reply.output_format or 'png'}", usage)


async def _gemini_image(llm: "LLM", candidate: Model, prompt: str, images: Sequence[Image]) -> Media:
    parents = [{"type": "image_url", "image_url": {"url": uri(p.mime, p.data)}} for p in images]
    reply = await _gemini(llm, candidate, [{"type": "text", "text": prompt}, *parents], "IMAGE")
    if not reply.choices or not reply.choices[0].message.images:  # gemini's NO_IMAGE leaves no choice at all
        why = reply.choices[0].finish_reason if reply.choices else "no answer"
        raise MissingMedia(f"{candidate.provider}/{candidate.model} answered without an image ({why})")
    data, mime = decode(reply.choices[0].message.images[0].image_url.url)
    return _made(candidate, data, mime, completion_usage(reply.usage))


async def _openai_speech(llm: "LLM", candidate: Model, text: str) -> Media:
    # a voice is required; only the streamed format reports usage, in its last event
    request = _request(candidate, {"model": candidate.model, "input": text, "voice": "alloy", "stream_format": "sse"})
    audio = await llm.client("openai").aspeech(**request)
    usage = None
    if request["stream_format"] == "sse":  # else extra asked for the plain format: the reply is the audio itself
        events = [e.json() for e in SSEDecoder().iter_bytes(iter([audio])) if e.data != "[DONE]"]
        done = next((e for e in events if e["type"] == "speech.audio.done"), None)
        if done is None:
            raise ProviderError("the speech stream ended before it finished", provider_name="openai")
        audio = b"".join(base64.b64decode(e["audio"]) for e in events if e["type"] == "speech.audio.delta")
        usage = Usage(input=done["usage"]["input_tokens"], output=done["usage"]["output_tokens"])
    if not audio:
        raise MissingMedia(f"openai/{candidate.model} answered without audio")
    return _made(candidate, audio, f"audio/{request.get('response_format', 'mp3')}", usage)


async def _gemini_speech(llm: "LLM", candidate: Model, text: str) -> Media:
    reply = await _gemini(llm, candidate, text, "AUDIO")
    audio = reply.choices[0].message.audio if reply.choices else None
    if not audio or not audio.data:
        raise MissingMedia(f"{candidate.provider}/{candidate.model} answered without audio")
    # any-llm wraps gemini's raw PCM in a WAV
    return _made(candidate, base64.b64decode(audio.data), "audio/wav", completion_usage(reply.usage))


async def _gemini(llm: "LLM", candidate: Model, content: str | list[dict], modality: str) -> ChatCompletion:
    request = _request(candidate, {"model": candidate.model, "messages": [{"role": "user", "content": content}],
                                   "response_modalities": [modality]})
    return cast(ChatCompletion, await llm.client(candidate.provider).acompletion(**request))  # not streamed

