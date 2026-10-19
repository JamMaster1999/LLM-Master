"""Real models driving every tool through the loop, on Modal sandboxes: each model gets tasks that need
several tools, and every model calls every tool. The checks are what the user would see."""
import io
import json
import uuid
import zipfile
from pathlib import Path

import pytest
from PIL import Image as Pillow

from agent import UserFile, media
from agent.sandbox import delete
from agent.turn import Conversation, turn
from server.app import MODELS

pytestmark = [pytest.mark.live, pytest.mark.parametrize("model", list(MODELS))]

FIXTURES = Path(__file__).parent.parent / "fixtures"
SKILLS = Path(__file__).parent / "skills"


class Play:
    """One conversation with one model, on a sandbox of its own."""

    def __init__(self, llm, model: str) -> None:
        self.llm, self.id = llm, f"test-{uuid.uuid4().hex[:12]}"
        candidate = MODELS[model]
        self.model = candidate.model_copy(update={"config": candidate.config.model_copy(
            update={"system": (FIXTURES / "prompt.md").read_text()})})
        self.conversation = Conversation()
        self.chunks: list[dict | UserFile] = []

    async def send(self, **kwargs) -> list[dict | UserFile]:
        sent = [chunk async for chunk in turn(self.llm, self.model, self.id, self.conversation, skills=SKILLS, **kwargs)]
        self.chunks += sent
        return sent

    def calls(self) -> list[str]:
        return [c["toolName"] for c in self.chunks if isinstance(c, dict) and c["type"] == "tool-input-available"]

    def files(self) -> dict[str, UserFile]:
        return {c.name: c for c in self.chunks if isinstance(c, UserFile)}

    def answer(self) -> str:
        """The last step's text, where the model answers."""
        chunks = [c for c in self.chunks if isinstance(c, dict)]
        last = max(i for i, c in enumerate(chunks) if c["type"] == "start-step")
        return "".join(c["delta"] for c in chunks[last:] if c["type"] == "text-delta")

    def trace(self) -> str:
        """Each call and its result, for a failure's message."""
        lines = []
        for c in self.chunks:
            if isinstance(c, UserFile):
                lines.append(f"  file {c.name} ({c.mime}, {len(c.data)} bytes)")
            elif c["type"] == "tool-input-available":
                lines.append(f"→ {c['toolName']} {json.dumps(c['input'])[:300]}")
            elif c["type"] in ("tool-output-available", "tool-output-error"):
                lines.append(f"{'←' if c['type'] == 'tool-output-available' else '✗'} "
                             f"{(c.get('output') or c.get('errorText') or '')[:300]}")
        return "\n".join(lines) + f"\nanswer: {self.answer()[:500]}"


@pytest.fixture
async def play(llm, model, monkeypatch):
    cheap = [m.model_copy(update={"extra": {"quality": "low", "size": "1024x1024"}}) for m in media.IMAGE_MODELS]
    monkeypatch.setattr(media, "IMAGE_MODELS", cheap)
    play = Play(llm, model)
    yield play
    await delete(play.id)


async def test_files_are_written_edited_read_and_delivered(play):
    await play.send(text="Do these steps in order. 1) Make a plan with the todo tool. 2) With write, create facts.md "
                         "holding three lines: 'Water boils at 90 C.', 'Ice melts at 0 C.', 'Steam is invisible.' "
                         "3) With edit, change 90 to 100. 4) Read facts.md back with read. 5) Count its lines with "
                         "bash, using wc -l. 6) Send facts.md with deliver_file.")
    assert {"todo", "write", "edit", "read", "bash", "deliver_file"} <= set(play.calls()), play.trace()
    delivered = play.files()["facts.md"].data.decode()
    assert "100 C" in delivered and "90 C" not in delivered, play.trace()


async def test_a_document_is_reviewed_before_it_is_delivered(play):
    await play.send(text="Write a short handout on the water cycle as handout.md, with three headings. Convert it "
                         "to handout.docx with pandoc, using bash. Check handout.docx with review_file, then send it "
                         "with deliver_file.")
    calls = play.calls()
    assert "review_file" in calls and calls.index("review_file") < calls.index("deliver_file"), play.trace()
    files = play.files()
    with zipfile.ZipFile(io.BytesIO(files["handout.docx"].data)) as docx:  # it opens, with the three headings
        assert docx.read("word/document.xml").decode().count('w:val="Heading') >= 3, play.trace()
    assert files["handout.pdf"].data.startswith(b"%PDF"), play.trace()


async def test_a_researched_answer_cites_its_sources(play):
    await play.send(text="1) Use web_search to find how high the boiling point of water is at the top of Mount "
                         "Everest. 2) Use run_program to run two web_search calls in parallel, one on why water "
                         "boils at a lower temperature at altitude and one on pressure cookers, and print the first "
                         "200 characters of each. 3) Read the most useful page in full with fetch_url. 4) Answer in "
                         "two sentences, citing your sources as [n].")
    assert {"web_search", "run_program", "fetch_url"} <= set(play.calls()), play.trace()
    sent = {c["sourceId"] for c in play.chunks if isinstance(c, dict) and c["type"] == "source-url"}
    assert any(f"[{n}]" in play.answer() for n in sent), play.trace()


async def test_an_image_and_speech_reach_the_user(play):
    await play.send(text="Make a simple image of a red apple on a desk with generate_image. Then use generate_audio "
                         "to say: Good morning, class.")
    files = [c for c in play.chunks if isinstance(c, UserFile)]  # each once: the user has them on arrival
    assert {"generate_image", "generate_audio"} <= set(play.calls()), play.trace()
    [picture] = [f for f in files if f.mime.startswith("image/")]
    [speech] = [f for f in files if f.mime.startswith("audio/")]
    assert Pillow.open(io.BytesIO(picture.data)).size[0] > 0 and len(speech.data) > 1000, play.trace()


async def test_the_users_answer_picks_what_the_skill_does(play):
    first = await play.send(text="Use ask_user to ask me which student to greet, with the options Ada and Grace. "
                                 "Then greet that student with the greet skill.")
    [asked] = [c for c in first if isinstance(c, dict) and c["type"] == "tool-input-available"
               and c["toolName"] == "ask_user"]
    question = asked["input"]["questions"][0]["question"]
    await play.send(answers={asked["toolCallId"]: {question: "Ada"}})
    assert "skill" in play.calls() and "Hello, Ada!" in play.answer(), play.trace()
