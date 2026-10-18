"""The message format: validation rules, per-API request goldens, and live acceptance of caller-written history."""
import base64
import io
import json
import os
import pathlib

import pytest
from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion
from any_llm.types.messages import MessageResponse
from PIL import Image as Pillow, ImageDraw, ImageFont

from llm_master.config import UnsupportedField
from llm_master.io.messages import Message, assemble, validate
from llm_master.io.parts import File, Image, Text
from llm_master.providers import API, ProviderName

GOLDENS = pathlib.Path(__file__).parent / "goldens"
PNG = {"type": "image", "data": "iVBORw0KGgo=", "mime": "image/png"}
PDF = {"type": "file", "data": "JVBERi0xLjQ=", "mime": "application/pdf", "filename": "a.pdf"}
GEMINI: tuple[ProviderName, str] = ("gemini", "gemini-3.5-flash")
CLAUDE: tuple[ProviderName, str] = ("anthropic", "claude-haiku-5-5")
GPT: tuple[ProviderName, str] = ("openai", "gpt-5.6-luna")


def origin(candidate):
    return {"provider": candidate[0], "model": candidate[1]}


# Every message kind once: each model's own turn (replayed to it, readable to the others), parallel
# calls, parts in user and tool messages, an empty result, a caller-written turn, a truncated turn.
HISTORY = [
    {"role": "user", "content": [{"type": "text", "text": "Weather in Paris? And what's in these?"}, PNG, PDF]},
    {"role": "assistant", "calls": [{"id": "lm_gem1", "name": "weather", "arguments": {"city": "Paris"},
                                     "wire_id": "call_g1"}],
     "reasoning": "Need the weather.", "origin": origin(GEMINI),
     "native": {"role": "assistant", "reasoning": "Need the weather.", "tool_calls": [
         {"id": "call_g1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\": \"Paris\"}"},
          "extra_content": {"google": {"thought_signature": "c2ln"}}}]}},
    {"role": "tool", "call_id": "lm_gem1", "content": [{"type": "text", "text": "18C, sunny"}, PNG]},
    {"role": "assistant", "content": "Checking two more.", "reasoning": "Two cities left.", "origin": origin(CLAUDE),
     "calls": [{"id": "lm_cla1", "name": "weather", "arguments": {"city": "Rome"}, "wire_id": "toolu_1"},
               {"id": "lm_cla2", "name": "weather", "arguments": {"city": "Oslo"}, "wire_id": "toolu_2"}],
     "native": [{"type": "thinking", "thinking": "Two cities left.", "signature": "sig"},
                {"type": "text", "text": "Checking two more."},
                {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Rome"}},
                {"type": "tool_use", "id": "toolu_2", "name": "weather", "input": {"city": "Oslo"}}]},
    {"role": "tool", "call_id": "lm_cla1", "content": "rain"},
    {"role": "tool", "call_id": "lm_cla2", "content": ""},
    {"role": "assistant", "origin": origin(GPT),
     "calls": [{"id": "lm_gpt1", "name": "read", "arguments": {"path": "a.pdf"}, "wire_id": "call_o1"}],
     "native": [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc1"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_o1", "name": "read",
                 "arguments": "{\"path\": \"a.pdf\"}", "status": "completed"}]},
    {"role": "tool", "call_id": "lm_gpt1", "content": [PDF]},
    {"role": "assistant", "origin": origin(GPT),
     "native": [{"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc2"}]},
    {"role": "user", "content": "Go on."},
    {"role": "assistant", "content": "All four are checked."},
    {"role": "user", "content": "Summarize."},
]


def reloaded(history) -> list[Message]:
    """Save to JSON and load back, as a caller's database would."""
    saved = json.dumps([m.model_dump(mode="json", exclude_none=True) for m in validate(history)])
    return validate(json.loads(saved))


@pytest.mark.parametrize("candidate, api, system", [(GEMINI, "completion", "Be brief."), (CLAUDE, "messages", None),
                                                    (GPT, "responses", None)])
def test_request_shape_survives_save_and_reload(candidate, api, system):
    provider, model = candidate
    expected = assemble(validate(HISTORY), provider, model, api, system)
    assert assemble(reloaded(HISTORY), provider, model, api, system) == expected
    golden = GOLDENS / f"{api}.json"
    if os.environ.get("GOLDEN_UPDATE"):
        golden.write_text(json.dumps(expected, indent=1) + "\n")
    assert expected == json.loads(golden.read_text()), f"{api} request changed: every saved conversation's cache busts"


def test_formats_without_a_spelling_raise():
    audio = [{"role": "user", "content": [{"type": "audio", "data": "UklGRg==", "format": "wav"}]}]
    assert assemble(validate(audio), "openai", "gpt-audio", "completion", None)["messages"][0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]
    with pytest.raises(UnsupportedField):
        assemble(validate(audio), *CLAUDE, "messages", None)
    with pytest.raises(UnsupportedField):
        assemble(validate(audio), *GPT, "responses", None)
    with pytest.raises(UnsupportedField, match="text-only"):
        assemble(validate(HISTORY[:3] + [{"role": "user", "content": "x"}]), "fireworks", "kimi-k3", "completion", None)


def test_reasoning_accepts_any_llms_object_form():
    m = Message.model_validate({"role": "assistant", "content": "x", "reasoning": {"content": "thought"}})
    assert m.reasoning == "thought"


U = {"role": "user", "content": "hi"}
CALL, CALL2 = {"id": "lm_1", "name": "f", "arguments": {}}, {"id": "lm_2", "name": "f", "arguments": {}}
ASKS = {"role": "assistant", "calls": [CALL]}
ANSWER = {"role": "tool", "call_id": "lm_1", "content": "ok"}


@pytest.mark.parametrize("history, error", [
    ([], "can't be empty"),
    ([{"role": "system", "content": "x"}, U], "'user', 'assistant' or 'tool'"),
    ([U, {"role": "assistant", "content": "x"}], "end on an assistant"),
    ([{"role": "tool", "content": "x"}], "call_id is required"),
    ([{**U, "call_id": "lm_1"}], "call_id is required"),
    ([{**U, "reasoning": "r"}], "can't carry reasoning"),
    ([U, {"role": "assistant", "content": "x", "native": {}}, U], "native requires origin"),
    ([U, {"role": "assistant"}, U], "needs content, calls or native"),
    ([U, {"role": "assistant", "content": [{"type": "text", "text": "x"}]}, U], "assistant content is a string"),
    ([U, ASKS, {**ANSWER, "content": [{"type": "audio", "data": "x", "format": "wav"}]}], "audio is user-only"),
    ([{**U, "contnet": "typo"}], "Extra inputs"),
    ([{"role": "user", "content": [{"type": "video", "data": "x"}]}], "does not match any of the expected tags"),
    ([U, {"role": "assistant", "calls": [{**CALL, "arguments": "{}"}]}], "valid dictionary"),
    ([U, {"role": "assistant", "calls": [{**CALL, "id": "has space"}]}], "should match pattern"),
    ([U, ASKS, ANSWER, U, ASKS, ANSWER], "'lm_1' repeats"),
    ([U, ASKS, {**ANSWER, "call_id": "lm_2"}], "answers no open call"),
    ([U, ASKS, U], "unanswered before the next user"),
    ([U, {"role": "assistant", "calls": [CALL, CALL2]}, ANSWER], r"\['lm_2'\] are unanswered"),
])
def test_rule_breaks_fail_before_sending(history, error):
    with pytest.raises(ValueError, match=error):
        validate(history)


# --- live: a caller-written tool loop with media, sent through each API's spelling ---

def page(text: str, fmt: str) -> str:
    img = Pillow.new("RGB", (520, 160), "white")
    ImageDraw.Draw(img).text((30, 50), text, fill="black", font=ImageFont.load_default(size=28))
    out = io.BytesIO()
    img.save(out, fmt)
    return f"data:{'image/png' if fmt == 'PNG' else 'application/pdf'};base64,{base64.b64encode(out.getvalue()).decode()}"


LOOKUP = {"name": "lookup", "description": "Look up a code",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}


def media_loop(tool_result) -> list[Message]:
    """Calls from 'another model'; the result carries the first code, the user's PDF the second."""
    return validate([
        {"role": "user", "content": "Use lookup twice in parallel: q='first' and q='other'."},
        {"role": "assistant", "content": "Looking both up.", "calls": [
            {"id": "lm_aaaaaaaaaaaaaaaaaaaaaaaa", "name": "lookup", "arguments": {"q": "first"}},
            {"id": "lm_bbbbbbbbbbbbbbbbbbbbbbbb", "name": "lookup", "arguments": {"q": "other"}}]},
        {"role": "tool", "call_id": "lm_aaaaaaaaaaaaaaaaaaaaaaaa", "content": [Text(text="Result attached."), tool_result]},
        {"role": "tool", "call_id": "lm_bbbbbbbbbbbbbbbbbbbbbbbb", "content": "nothing found"},
        {"role": "user", "content": [Text(text="Reply with the looked-up code, then the code in this PDF."),
                                     File(page("PDF CODE: OTTER-3302", "PDF"), filename="code.pdf")]},
    ])


async def answer(provider: ProviderName, model: str, api: API, history: list[Message]) -> str:
    client = AnyLLM.create(provider, unified_exceptions=True, max_retries=0)
    request = assemble(history, provider, model, api, None)
    if api == "completion":
        r = await client.acompletion(model=model, tools=[{"type": "function", "function": LOOKUP}], **request)
        assert isinstance(r, ChatCompletion)
        return r.choices[0].message.content or ""
    if api == "messages":
        r = await client.amessages(model=model, max_tokens=500, **request, tools=[
            {"name": "lookup", "description": LOOKUP["description"], "input_schema": LOOKUP["parameters"]}])
        assert isinstance(r, MessageResponse)
        return "".join(b.text for b in r.content if b.type == "text")
    r = await client.aresponses(model=model, tools=[{"type": "function", **LOOKUP}], **request)
    return "".join(p.text for i in r.output if i.type == "message" for p in i.content if p.type == "output_text")


@pytest.mark.live
@pytest.mark.parametrize("provider, model, api", [("anthropic", "claude-haiku-5-5", "messages"),
                                                  ("openai", "gpt-5.5", "responses"),
                                                  ("gemini", "gemini-3.5-flash", "completion")])
@pytest.mark.parametrize("fmt", ["PNG", "PDF"])
async def test_media_in_tool_results_is_read(provider, model, api, fmt):
    result = Image(page("TOOL CODE: ZEBRA-7741", fmt)) if fmt == "PNG" else File(page("TOOL CODE: ZEBRA-7741", fmt))
    text = await answer(provider, model, api, media_loop(result))
    assert "ZEBRA-7741" in text and "OTTER-3302" in text, text


@pytest.mark.live
async def test_fireworks_reads_text_results_of_other_models_calls():
    history = media_loop(Text(text="TOOL CODE: ZEBRA-7741"))
    history[-1] = Message(role="user", content="Reply with the looked-up code only.")
    text = await answer("fireworks", "accounts/fireworks/models/kimi-k3", "completion", history)
    assert "ZEBRA-7741" in text, text
