"""image() and speech() at their entry points, against recorded provider exchanges (RECORD=1 re-records)."""
import base64
import io

import pytest
from PIL import Image as Pillow

from llm_master.config import UnsupportedField
from llm_master.facade import LLM
from llm_master.io.parts import Image
from llm_master.media import Media, MissingMedia
from llm_master.types import Config, Model

# small outputs keep the recordings small
OPENAI = Model(provider="openai", model="gpt-image-2",
               extra={"size": "1024x1024", "quality": "low", "output_format": "jpeg", "output_compression": 20})
GEMINI = Model(provider="gemini", model="gemini-3.1-flash-image", extra={"image_config": {"image_size": "512"}})
DRAW = "A small red circle on a white background, flat."


@pytest.fixture
async def llm():
    llm = LLM()
    yield llm
    await llm.aclose()


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
    recorded(f"{candidate.provider}_edit")  # gemini's request body pins the parent; openai's multipart pins its path
    edited = await llm.image(candidate, "Make the square blue.", [parent()])
    assert kind(edited) == "JPEG"


@pytest.mark.parametrize("candidate, mime, reports_usage", [
    (Model(provider="openai", model="gpt-4o-mini-tts"), "audio/mpeg", False),
    (Model(provider="gemini", model="gemini-3.8-flash-tts"), "audio/wav", True),
], ids=["openai", "gemini"])
async def test_speech_comes_back_as_audio(candidate, mime, reports_usage, recorded, llm):
    recorded(f"{candidate.provider}_speech")
    spoken = await llm.speech(candidate, "Say: hello there.")
    assert spoken.mime == mime and len(spoken.data) > 1000
    assert (spoken.usage is not None) == reports_usage and spoken.cost is None


async def test_an_answer_without_the_image_raises_typed(recorded, llm):
    recorded("gemini_no_image")  # gemini-3-pro-image answers content_filter with no image (live, 2026-10-18)
    with pytest.raises(MissingMedia, match="content_filter"):
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
