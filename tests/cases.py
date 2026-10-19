"""Candidates and inputs that several test modules send. A recorded fixture pins each request they build."""
import base64
import io
import pathlib

from PIL import Image as Pillow

from llm_master.config.tool_choice import ToolChoice
from llm_master.io.parts import Image
from llm_master.types import Config, Model

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
WRITE = {"type": "function", "function": {"name": "write_file", "description": "Write a file",
         "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}}
ASK = [{"role": "user", "content": "Think carefully: a farmer has 17 sheep; all but 9 run away, then he buys twice as "
                                   "many as remain and sells a third of the flock. How many sheep now? Then call "
                                   "get_weather for Toronto and for Paris, in parallel."}]
LOOPS = {
    "anthropic": Model(provider="anthropic", model="claude-sonnet-5-5", config=Config(tools=[WEATHER], thinking="high")),
    "openai": Model(provider="openai", model="gpt-5.6-luna", config=Config(tools=[WEATHER], thinking="high")),
    "gemini": Model(provider="gemini", model="gemini-3.5-flash", config=Config(tools=[WEATHER], thinking="low")),
    "fireworks": Model(provider="fireworks", model="accounts/fireworks/models/deepseek-v4p1-flash",
                       config=Config(tools=[WEATHER])),
}
GREET = [{"role": "user", "content": "Say good morning to Toronto."}]  # CONFIGURED's recordings ask this
CONFIGURED = {  # each field placed per API; the recorded request body pins the spelling
    "anthropic": Model(provider="anthropic", model="claude-haiku-5-5", config=Config(
        system="Answer in French.", thinking="low", max_tokens=2000, tools=[WEATHER],
        tool_choice=ToolChoice(mode="none"))),
    "openai": Model(provider="openai", model="gpt-5.6-luna", config=Config(
        system="Answer in French.", thinking="low", max_tokens=2000, tools=[WEATHER],
        tool_choice=ToolChoice(mode="none"))),
    "gemini": Model(provider="gemini", model="gemini-3.5-flash", config=Config(
        system="Answer in French.", thinking="low", max_tokens=2000, cache=False, temperature=0.5, tools=[WEATHER],
        tool_choice=ToolChoice(mode="none"))),
    "openai-none": Model(provider="openai", model="gpt-5.6-luna", config=Config(system="Answer in French.",
                                                                                 cache=False)),
}
CHAIN = [CONFIGURED["anthropic"], CONFIGURED["openai"]]  # a fallback pair; both recordings ask GREET
# the streamed anthropic call is the sly one: eager_input_streaming sends the arguments unbuffered, and
# anthropic's SDK parses the cut JSON into a valid-looking input
CUT = {"openai": Model(provider="openai", model="gpt-5.6-luna",
                       config=Config(tools=[WRITE], thinking="none", max_tokens=16)),
       "anthropic": Model(provider="anthropic", model="claude-haiku-5-5", config=Config(
           tools=[{**WRITE, "function": {**WRITE["function"], "eager_input_streaming": True}}],
           tool_choice=ToolChoice(mode="required"), thinking="none", max_tokens=60))}
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
