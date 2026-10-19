"""image() and speech() at their entry points, against recorded provider exchanges (RECORD=1 re-records)."""
import base64
import io
import wave

import pytest
from cases import DRAW, GEMINI_IMAGE, OPENAI_IMAGE, parent
from PIL import Image as Pillow

from llm_master.config import UnsupportedField
from llm_master.io.parts import Image
from llm_master.media import Media, MissingMedia
from llm_master.types import Config, Model


def kind(media: Media) -> str:
    return Pillow.open(io.BytesIO(media.data)).format or ""


@pytest.mark.parametrize("candidate", [OPENAI_IMAGE, GEMINI_IMAGE], ids=lambda m: m.provider)
async def test_an_image_comes_back_as_bytes(candidate, recorded, llm):
    recorded(f"{candidate.provider}_image")
    made = await llm.image(candidate, DRAW)
    assert (made.mime, kind(made)) == ("image/jpeg", "JPEG")
    assert made.origin.model == candidate.model and made.usage and made.usage.output > 0


@pytest.mark.parametrize("candidate", [OPENAI_IMAGE, GEMINI_IMAGE], ids=lambda m: m.provider)
async def test_an_edit_sends_the_parent_with_the_instruction(candidate, recorded, llm):
    recorded(f"{candidate.provider}_edit")  # the recorded request body pins the parent and the instruction
    edited = await llm.image(candidate, "Make the square blue.", [parent()])
    assert kind(edited) == "JPEG"


async def test_extra_has_the_last_word(recorded, llm):
    recorded("openai_speech_plain")  # the request carries the caller's plain format, not our streamed one
    spoken = await llm.speech(Model(provider="openai", model="gpt-4o-mini-tts", extra={"stream_format": "audio"}),
                              "Say: hello there.")
    assert spoken.data and spoken.usage is None  # the plain format reports no usage


async def test_openai_speech_reports_its_usage(recorded, llm):
    recorded("openai_speech")  # the streamed format: the plain one carries no usage
    spoken = await llm.speech(Model(provider="openai", model="gpt-4o-mini-tts"), "Say: hello there.")
    assert spoken.mime == "audio/mp3" and spoken.data and spoken.usage and spoken.usage.output > 0


async def test_gemini_speech_is_a_wav(recorded, llm):
    recorded("gemini_speech")
    spoken = await llm.speech(Model(provider="gemini", model="gemini-3.8-flash-tts"), "Say: hello there.")
    assert spoken.mime == "audio/wav" and wave.open(io.BytesIO(spoken.data)).getnframes() > 0
    assert spoken.usage and spoken.usage.output > 0


@pytest.mark.parametrize("name, why", [
    ("gemini_no_image", "content_filter"),  # gemini-3-pro-image answers IMAGE_RECITATION (live, 2026-10-18)
    ("gemini_no_image_choice", "no answer"),  # the same reply as NO_IMAGE, which any-llm turns into no choice
])
async def test_an_answer_without_the_image_raises_typed(name, why, recorded, llm):
    recorded(name)
    with pytest.raises(MissingMedia, match=why):
        await llm.image(Model(provider="gemini", model="gemini-3-pro-image"), DRAW)


@pytest.mark.parametrize("candidate", [Model(provider="anthropic", model="claude-haiku-5-5"),
                                       Model(provider="openai", model="gpt-image-2", config=Config(system="x")),
                                       Model(provider="openai", model="gpt-image-2", extra={"n": 2})],
                         ids=["no-image-api", "conversation-config", "two-images"])
async def test_a_media_call_that_cant_be_made_sends_nothing(candidate, recorded, llm):
    recorded("nothing_sent")
    with pytest.raises(UnsupportedField):
        await llm.image(candidate, DRAW)


@pytest.mark.live
async def test_one_providers_image_edits_on_another(llm):
    made = await llm.image(GEMINI_IMAGE, DRAW)
    source = Image(f"data:{made.mime};base64,{base64.b64encode(made.data).decode()}")
    edited = await llm.image(OPENAI_IMAGE, "Make the circle blue.", [source])
    assert kind(edited) == "JPEG" and edited.origin.provider == "openai"
