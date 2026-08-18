"""response_format: Json goldens, live structured output on all three APIs."""
import json
from typing import Literal

import pytest
from any_llm import AnyLLM
from pydantic import BaseModel, ValidationError

from llm_master.config import UnsupportedField, response_format, thinking
from llm_master.config.response_format import Format


class Verdict(BaseModel):
    answer: int
    confidence: Literal["low", "high"]


PROMPT = [{"role": "user", "content": "What is 2+2? Give your confidence."}]


def test_schema_is_one_rename():
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    assert response_format.translate(Format(type="schema", schema=Verdict), "completion") == {"response_format": Verdict}
    assert response_format.translate(Format(type="schema", schema=schema), "responses") == {"response_format": schema}
    assert response_format.translate(Format(type="schema", schema=Verdict), "messages") == {"output_format": Verdict}


def test_bare_json_mode():
    assert response_format.translate(Format(type="json"), "completion") == {"response_format": {"type": "json_object"}}
    assert response_format.translate(Format(type="json"), "responses") == {"response_format": {"type": "json_object"}}
    with pytest.raises(UnsupportedField):
        response_format.translate(Format(type="json"), "messages")


def test_text_emits_nothing_and_contradictions_raise():
    for api in ("completion", "messages", "responses"):
        assert response_format.translate(Format(type="text"), api) == {}
    with pytest.raises(ValidationError):
        Format(type="schema")  # needs a schema
    with pytest.raises(ValidationError):
        Format(type="json", schema=Verdict)  # schema only goes with type="schema"


JSON_PROMPT = [{"role": "user", "content": "List the first two primes as JSON with a single key 'primes'."}]


@pytest.mark.live
async def test_completion_json_mode():
    llm = AnyLLM.create("gemini")
    r = await llm.acompletion(model="gemini-2.5-flash", messages=JSON_PROMPT, max_tokens=500,
                              **thinking.translate("none", "completion"),
                              **response_format.translate(Format(type="json"), "completion"))
    # json mode guarantees syntax, not value types (live: primes sometimes arrive as strings)
    assert "primes" in json.loads(r.choices[0].message.content)


@pytest.mark.live
async def test_responses_json_mode():
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", JSON_PROMPT,
                             **thinking.translate("none", "responses"),
                             **response_format.translate(Format(type="json"), "responses"))
    text = "".join(p.text for i in r.output if i.type == "message" for p in i.content if p.type == "output_text")
    assert "primes" in json.loads(text)


def _as_verdict(obj) -> Verdict:
    if isinstance(obj, Verdict):
        return obj
    return Verdict.model_validate(obj if isinstance(obj, dict) else json.loads(obj))


@pytest.mark.live
async def test_completion_structured():
    llm = AnyLLM.create("gemini")
    r = await llm.acompletion(model="gemini-2.5-flash", messages=PROMPT, max_tokens=500,
                              **thinking.translate("none", "completion"),
                              **response_format.translate(Format(type="schema", schema=Verdict), "completion"))
    msg = r.choices[0].message
    v = _as_verdict(getattr(msg, "parsed", None) or msg.content)
    assert v.answer == 4


@pytest.mark.live
async def test_messages_structured():
    llm = AnyLLM.create("anthropic")
    r = await llm.amessages("claude-sonnet-5", PROMPT, 500,
                            **thinking.translate("none", "messages"),
                            **response_format.translate(Format(type="schema", schema=Verdict), "messages"))
    v = _as_verdict(getattr(r, "parsed", None) or "".join(b.text for b in r.content if b.type == "text"))
    assert v.answer == 4


@pytest.mark.live
async def test_responses_structured():
    llm = AnyLLM.create("openai")
    r = await llm.aresponses("gpt-5.1", PROMPT,
                             **thinking.translate("none", "responses"),
                             **response_format.translate(Format(type="schema", schema=Verdict), "responses"))
    parsed = getattr(r, "output_parsed", None)
    v = _as_verdict(parsed if parsed is not None else
                    "".join(p.text for i in r.output if i.type == "message"
                            for p in i.content if p.type == "output_text"))
    assert v.answer == 4
