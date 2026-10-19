"""Candidates and inputs that several test modules send. A recorded fixture pins each request they build."""
import base64
import io
import pathlib

from PIL import Image as Pillow

from llm_master.io.parts import Image
from llm_master.types import Config, Model

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
LONG = "Remember these facts. " + " ".join(f"Fact {i}: the code for item {i} is {i * 7919 % 10007}." for i in range(700))

# small outputs keep the recordings small; an explicit None config field means unset, as everywhere
OPENAI_IMAGE = Model(provider="openai", model="gpt-image-2", config=Config(timeout=120, system=None),
                     extra={"size": "1024x1024", "quality": "low", "output_format": "jpeg", "output_compression": 20})
GEMINI_IMAGE = Model(provider="gemini", model="gemini-3.1-flash-image", extra={"image_config": {"image_size": "512"}})
DRAW = "A small red circle on a white background, flat."


def parent() -> Image:
    """A red square to edit."""
    out = io.BytesIO()
    Pillow.new("RGB", (64, 64), "red").save(out, "PNG")
    return Image("data:image/png;base64," + base64.b64encode(out.getvalue()).decode())
