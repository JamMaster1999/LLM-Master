"""The generate pair: listed only with llm, and live through Toolbox.run on Modal, with real or stand-in providers."""
import base64
import io
import json
import uuid

import pytest
from PIL import Image as Pillow

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
    # a BMP parent, which the edit endpoint doesn't take, is redrawn as JPEG on the way
    await box.run("bash", {"command": f"python -c \"from PIL import Image; Image.open('{saved(made)}').save('a.bmp')\""})
    edited = await box.run("generate_image", {"prompt": "Make the circle blue.", "image": "a.bmp"})
    candidate = media.IMAGE_MODELS[0].model_dump(include={"provider", "model"})
    assert await record(box, saved(made)) == {"parent": None, "prompt": DRAW, "candidate": candidate}
    assert await record(box, saved(edited)) == {"parent": "a.bmp", "prompt": "Make the circle blue.",
                                                "candidate": candidate}
    assert not (await box.run("deliver_file", {"path": saved(made)})).error  # the model saw it, so it's reviewed
    for result in (made, edited):  # the model sees the image; the user gets the file
        [file] = result.files
        assert saved(result) == f"images/{file.name}" and file.name.endswith("." + file.mime.split("/")[1])
        assert [part["type"] for part in result.content] == ["text", "image"]


@pytest.mark.live  # on Modal; a stand-in answers for OpenAI, to show what the edit sent
async def test_an_edit_sends_the_parent_with_the_prompt(provider_server, box):
    red, blue = Pillow.new("RGB", (64, 64), "red"), Pillow.new("RGB", (64, 64), "blue")
    photo, made = io.BytesIO(), io.BytesIO()
    red.save(photo, "MPO", save_all=True, append_images=[blue])  # a phone's multi-picture JPEG
    blue.save(made, "PNG")
    edits = []

    def answer(handler, provider, path, body):
        edits.append((path, body))
        reply = json.dumps({"created": 0, "data": [{"b64_json": base64.b64encode(made.getvalue()).decode()}]}).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(reply)))
        handler.end_headers()
        handler.wfile.write(reply)
    provider_server(answer)
    await box.run("bash", {"command": f"echo {base64.b64encode(photo.getvalue()).decode()} | base64 -d > photo.jpg"})
    saved(await box.run("generate_image", {"prompt": "Make the circle blue.", "image": "photo.jpg"}))
    [(path, body)] = edits
    assert path == "/images/edits" and b'name="prompt"\r\n\r\nMake the circle blue.\r\n' in body
    assert b"Content-Type: image/jpeg\r\n\r\n" + photo.getvalue() + b"\r\n" in body  # as it is, not redrawn


@pytest.mark.live
async def test_speech_is_saved_and_sent_to_the_user(box):
    spoken = await box.run("generate_audio", {"text": "Hello."})
    [file] = spoken.files
    assert spoken.content == f"Saved audio/{file.name} and sent it to the user."
    size = await box.run("bash", {"command": f"wc -c < audio/{file.name}"})
    assert (file.mime, size.content) == ("audio/mp3", str(len(file.data)))
