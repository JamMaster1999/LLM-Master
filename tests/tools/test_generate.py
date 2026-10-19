"""The generate pair: listed only with llm, and live through Toolbox.run on real providers and Modal."""
import json
import uuid

import pytest

from agent import Result, Toolbox, media
from agent.sandbox import delete
from llm_master.facade import LLM

DRAW = "A small red circle on a white background, flat."


@pytest.mark.parametrize("llm", [None, LLM()], ids=["without-llm", "with-llm"])
def test_the_generate_pair_comes_with_llm(llm):
    tools = {spec["function"]["name"]: spec["function"] for spec in Toolbox("c", llm=llm).specs}
    listed = llm is not None
    for name in ("generate_image", "generate_audio"):  # for the model, and for its programs
        assert (name in tools, name in tools["run_program"]["description"]) == (listed, listed), name


@pytest.fixture
async def box(llm, monkeypatch):
    cheap = [m.model_copy(update={"extra": {"quality": "low", "size": "1024x1024"}}) for m in media.IMAGE_MODELS]
    monkeypatch.setattr(media, "IMAGE_MODELS", cheap)  # the toolbox is under test here, not the art
    box = Toolbox(f"test-{uuid.uuid4().hex[:12]}", llm=llm)
    yield box
    await box.aclose()
    await delete(box.sandbox.conversation_id)


def saved(result: Result) -> str:
    assert not result.error, result.content
    return result.text().removeprefix("Saved ").removesuffix(" and sent it to the user.")


async def record(box: Toolbox, image: str) -> dict:
    return json.loads((await box.run("read", {"path": image.rsplit(".", 1)[0] + ".json"})).text())


@pytest.mark.live
async def test_an_image_is_made_then_edited_with_its_parent_on_record(box):
    made = await box.run("generate_image", {"prompt": DRAW})
    edited = await box.run("generate_image", {"prompt": "Make the circle blue.", "image": saved(made)})
    candidate = media.IMAGE_MODELS[0].model_dump(include={"provider", "model"})
    assert await record(box, saved(made)) == {"parent": None, "prompt": DRAW, "candidate": candidate}
    assert await record(box, saved(edited)) == {"parent": saved(made), "prompt": "Make the circle blue.",
                                                "candidate": candidate}
    for result in (made, edited):  # the model sees the image; the user gets the file
        [file] = result.files
        assert saved(result) == f"images/{file.name}" and file.name.endswith("." + file.mime.split("/")[1])
        assert [part["type"] for part in result.content] == ["text", "image"]


@pytest.mark.live
async def test_speech_is_saved_and_sent_to_the_user(box):
    spoken = await box.run("generate_audio", {"text": "Hello."})
    [file] = spoken.files
    assert spoken.content == f"Saved audio/{file.name} and sent it to the user."
    size = await box.run("bash", {"command": f"wc -c < audio/{file.name}"})
    assert (file.mime, size.content) == ("audio/mp3", str(len(file.data)))
