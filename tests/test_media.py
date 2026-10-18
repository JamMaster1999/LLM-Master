"""image() and speech() at their entry points, against recorded provider exchanges (RECORD=1 re-records)."""
import base64
import io
import wave

import pytest
from PIL import Image as Pillow

from llm_master.config import UnsupportedField
from llm_master.io.parts import Image
from llm_master.media import Media, MissingMedia
from llm_master.types import Config, Model

# small outputs keep the recordings small; an explicit None config field means unset, as everywhere
OPENAI = Model(provider="openai", model="gpt-image-2", config=Config(timeout=120, system=None),
               extra={"size": "1024x1024", "quality": "low", "output_format": "jpeg", "output_compression": 20})
GEMINI = Model(provider="gemini", model="gemini-3.1-flash-image", extra={"image_config": {"image_size": "512"}})
DRAW = "A small red circle on a white background, flat."


def kind(media: Media) -> str:
    return Pillow.open(io.BytesIO(media.data)).format or ""


def parent() -> Image:
    out = io.BytesIO()
    Pillow.new("RGB", (64, 64), "red").save(out, "PNG")
    return Image("data:image/png;base64," + base64.b64encode(out.getvalue()).decode())


@pytest.mark.parametrize("candidate", [OPENAI, GEMINI], ids=lambda m: m.provider)
async def test_an_image_comes_back_as_bytes(candidate, recorded, llm):
    recorded(f"{candidate.provider}_image")
    made = await llm.image(candidate, DRAW)
    assert (made.mime, kind(made)) == ("image/jpeg", "JPEG")
    assert made.origin.model == candidate.model and made.usage and made.usage.output > 0


@pytest.mark.parametrize("candidate", [OPENAI, GEMINI], ids=lambda m: m.provider)
async def test_an_edit_sends_the_parent_with_the_instruction(candidate, recorded, llm):
    recorded(f"{candidate.provider}_edit")  # the recorded request body pins the parent and the instruction
    edited = await llm.image(candidate, "Make the square blue.", [parent()])
    assert kind(edited) == "JPEG"


async def test_extra_has_the_last_word(recorded, llm):
    recorded("gemini_text_and_image")  # the request asks for TEXT and IMAGE, not our IMAGE alone
    candidate = GEMINI.model_copy(update={"extra": {**GEMINI.extra, "response_modalities": ["TEXT", "IMAGE"]}})
    assert kind(await llm.image(candidate, DRAW)) == "JPEG"


async def test_openai_speech_is_typed_by_its_reply_and_reports_no_usage(recorded, llm):
    recorded("openai_speech")
    spoken = await llm.speech(Model(provider="openai", model="gpt-4o-mini-tts"), "Say: hello there.")
    assert spoken.mime == "audio/mpeg" and spoken.usage is None  # so its cost stays None, never zero


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
                                       Model(provider="openai", model="gpt-image-2", config=Config(system="x"))],
                         ids=["no-image-api", "conversation-config"])
async def test_a_media_call_that_cant_be_made_sends_nothing(candidate, recorded, llm):
    recorded("nothing_sent")
    with pytest.raises(UnsupportedField):
        await llm.image(candidate, DRAW)


@pytest.mark.live
async def test_one_providers_image_edits_on_another(llm):
    made = await llm.image(GEMINI, DRAW)
    source = Image(f"data:{made.mime};base64,{base64.b64encode(made.data).decode()}")
    edited = await llm.image(OPENAI, "Make the circle blue.", [source])
    assert kind(edited) == "JPEG" and edited.origin.provider == "openai"
