"""The message format: validation rules, per-API request goldens, and live acceptance of caller-written history."""
import base64
import datetime
import io
import json
import os
import pathlib

import pytest
from PIL import Image as Pillow, ImageDraw, ImageFont

from llm_master.config import UnsupportedField
from llm_master.facade import LLM  # the type of conftest's llm fixture
from llm_master.io.messages import Message, assemble, validate
from llm_master.io.parts import File, Image, Text, uri
from llm_master.providers import ProviderName
from llm_master.types import Config, Model

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
def test_request_bytes_survive_save_and_reload(candidate, api, system):
    provider, model = candidate
    sent = json.dumps(assemble(validate(HISTORY), provider, model, api, system), indent=1) + "\n"
    assert json.dumps(assemble(reloaded(HISTORY), provider, model, api, system), indent=1) + "\n" == sent
    golden = GOLDENS / f"{api}.json"
    if os.environ.get("GOLDEN_UPDATE"):
        golden.write_text(sent)
    assert sent == golden.read_text(), f"{api} request changed: every saved conversation's cache busts"


AUDIO = [{"role": "user", "content": [{"type": "audio", "data": "UklGRg==", "format": "wav"}]}]


@pytest.mark.parametrize("history, provider, model, api, error", [
    (AUDIO, *CLAUDE, "messages", "no audio input"),
    (AUDIO, *GPT, "responses", "no audio input"),
], ids=["audio-messages", "audio-responses"])
def test_content_without_a_spelling_raises(history, provider, model, api, error):
    with pytest.raises(UnsupportedField, match=error):
        assemble(validate(history), provider, model, api, None)


def test_claudes_empty_answer_is_left_out_of_its_own_replay():
    empty = {"role": "assistant", "content": "", "origin": origin(CLAUDE), "native": []}
    sent = assemble(validate([U, empty, U]), *CLAUDE, "messages", None)["messages"]
    assert [m["role"] for m in sent] == ["user", "user"]


U = {"role": "user", "content": "hi"}
CALL, CALL2 = {"id": "lm_1", "name": "f", "arguments": {}}, {"id": "lm_2", "name": "f", "arguments": {}}
ASKS = {"role": "assistant", "calls": [CALL]}
ANSWER = {"role": "tool", "call_id": "lm_1", "content": "ok"}


@pytest.mark.parametrize("history, error", [  # our rules match our words, pydantic's its error type
    ([], "can't be empty"),
    ([{"role": "system", "content": "x"}, U], "literal_error"),
    ([{"role": "tool", "content": "x"}], "call_id is required"),
    ([{**U, "call_id": "lm_1"}], "call_id is required"),
    ([{**U, "reasoning": "r"}], "can't carry reasoning"),
    ([U, {"role": "assistant", "content": "x", "native": {}}, U], "native requires origin"),
    ([U, {"role": "assistant"}, U], "needs content, calls or native"),
    ([U, {"role": "assistant", "content": []}, U], "assistant content is a string"),
    ([U, ASKS, {**ANSWER, "content": [{"type": "audio", "data": "x", "format": "wav"}]}], "audio is user-only"),
    ([{**U, "contnet": "typo"}], "extra_forbidden"),
    ([{"role": "user", "content": [{"type": "video", "data": "x"}]}], "union_tag_invalid"),
    ([U, {"role": "assistant", "calls": [{**CALL, "arguments": "{}"}]}], "dict_type"),
    ([U, {"role": "assistant", "calls": [{**CALL, "arguments": {"on": datetime.date(2026, 1, 1)}}]}],
     "invalid-json-value"),
    ([U, {"role": "assistant", "calls": [{**CALL, "id": "has space"}]}], "string_pattern_mismatch"),
    ([U, ASKS, ANSWER, U, ASKS, ANSWER], "'lm_1' repeats"),
    ([U, ASKS, {**ANSWER, "call_id": "lm_2"}], "answers no open call"),
    ([U, ASKS, U], "unanswered before the next user"),
    ([U, {"role": "assistant", "calls": [CALL, CALL2]}, ANSWER], r"\['lm_2'\] are unanswered"),
])
def test_rule_breaks_fail_before_sending(history, error):
    with pytest.raises(ValueError, match=error):
        validate(history)


# --- live: a caller-written tool loop with media, sent through query() on each API ---

def page(text: str, fmt: str, mime: str) -> str:
    img = Pillow.new("RGB", (520, 160), "white")
    ImageDraw.Draw(img).text((30, 50), text, fill="black", font=ImageFont.load_default(size=28))
    out = io.BytesIO()
    img.save(out, fmt)
    return uri(mime, base64.b64encode(out.getvalue()).decode())


LOOKUP = [{"type": "function", "function": {"name": "lookup", "description": "Look up a code", "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]


def media_loop(tool_result, ask) -> list[Message]:
    """Calls from 'another model'; the result carries the first code, the last user message the ask."""
    return validate([
        {"role": "user", "content": "Use lookup twice in parallel: q='first' and q='other'."},
        {"role": "assistant", "content": "Looking both up.", "calls": [
            {"id": "lm_aaaaaaaaaaaaaaaaaaaaaaaa", "name": "lookup", "arguments": {"q": "first"}},
            {"id": "lm_bbbbbbbbbbbbbbbbbbbbbbbb", "name": "lookup", "arguments": {"q": "other"}}]},
        {"role": "tool", "call_id": "lm_aaaaaaaaaaaaaaaaaaaaaaaa", "content": [Text(text="Result attached."),
                                                                             tool_result]},
        {"role": "tool", "call_id": "lm_bbbbbbbbbbbbbbbbbbbbbbbb", "content": "nothing found"},
        {"role": "user", "content": ask},
    ])


async def answer(llm: LLM, provider: ProviderName, model: str, history: list[Message]) -> str:
    r = await llm.query(Model(provider=provider, model=model, config=Config(tools=LOOKUP, max_tokens=500)), history)
    return str(r.message.content)


@pytest.mark.live
@pytest.mark.parametrize("provider, model", [("anthropic", "claude-haiku-5-5"), ("openai", "gpt-5.5"),
                                             ("gemini", "gemini-3.5-flash")])
@pytest.mark.parametrize("make, fmt, mime", [(Image, "PNG", "image/png"), (File, "PDF", "application/pdf")],
                         ids=["image", "pdf"])
async def test_media_in_tool_results_is_read(provider, model, make, fmt, mime, llm):
    ask = [Text(text="Reply with the looked-up code, then the code in this PDF."),
           File(page("PDF CODE: OTTER-3302", "PDF", "application/pdf"), filename="code.pdf")]
    history = media_loop(make(page("TOOL CODE: ZEBRA-7741", fmt, mime)), ask)
    text = await answer(llm, provider, model, history)
    assert "ZEBRA-7741" in text and "OTTER-3302" in text, text


@pytest.mark.live
async def test_fireworks_reads_an_image_result_of_another_models_call(llm):
    result = Image(page("TOOL CODE: ZEBRA-7741", "PNG", "image/png"))
    history = media_loop(result, "Reply with the looked-up code only.")
    text = await answer(llm, "fireworks", "accounts/fireworks/models/kimi-k3", history)
    assert "ZEBRA-7741" in text, text
