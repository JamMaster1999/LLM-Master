"""query() at its entry point, against recorded provider exchanges (RECORD=1 re-records them live)."""
import asyncio
import json
import pathlib
import threading
import time
from typing import Literal

import pytest
from any_llm.exceptions import ProviderError
from pydantic import BaseModel, Field

from llm_master.config import UnsupportedField
from llm_master.config.moderation import Flagged
from llm_master.config.response_format import Format
from llm_master.config.tool_choice import ToolChoice
from llm_master.facade import LLM
from llm_master.io.messages import validate
from llm_master.io.parts import Audio, Text
from llm_master.io.response import CutCall
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


async def ask(llm: LLM, candidate: Model, history: list, stream: bool):
    if not stream:
        return await llm.query(candidate, history), []
    s = await llm.query(candidate, history, stream=True)
    chunks = [c async for c in s]
    assert s.response is not None
    return s.response, chunks


def saved(history: list) -> list[dict]:
    """Through JSON and back, as a caller's database would."""
    return json.loads(json.dumps([m.model_dump(mode="json", exclude_none=True) for m in validate(history)]))


def check_stream(chunks: list[dict], response) -> None:
    """The chunks tell the same answer as stream.response, under the same call ids."""
    assert chunks[0] == {"type": "start-step"} and chunks[-1] == {"type": "finish-step"}
    assert "".join(c["delta"] for c in chunks if c["type"] == "text-delta") == response.message.content
    for block in {c["id"] for c in chunks if c["type"] == "reasoning-start"}:
        assert "".join(c["delta"] for c in chunks if c.get("id") == block and c["type"] == "reasoning-delta") \
            in (response.message.reasoning or "")
    calls = response.message.calls or []
    assert [c["toolCallId"] for c in chunks if c["type"] == "tool-input-start"] == [c.id for c in calls]
    assert {c["toolCallId"]: c["input"] for c in chunks if c["type"] == "tool-input-available"} == \
        {c.id: c.arguments for c in calls}


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("provider", LOOPS)
async def test_tool_loop_replays_the_models_own_turn(provider, stream, recorded, llm):
    recorded(f"{provider}_loop" + ("_stream" if stream else ""))
    candidate = LOOPS[provider]
    first, chunks = await ask(llm, candidate, ASK, stream)
    calls = first.message.calls or []
    assert first.finish == "calls" and sorted(str(c.arguments["city"]) for c in calls) == ["Paris", "Toronto"]
    assert all(c.id.startswith("lm_") and c.wire_id for c in calls)
    assert first.message.origin and first.message.origin.model == candidate.model and first.message.native
    assert first.usage and first.message.reasoning and first.usage.reasoning > 0
    assert first.usage.input > 0 and first.usage.output > 0 and len(first.attempts) == 1
    if stream:
        check_stream(chunks, first)
    results = [{"role": "tool", "call_id": c.id, "content": f"{c.arguments['city']}: 20C, sunny"} for c in calls]
    second, chunks = await ask(llm, candidate, saved(ASK + [first.message] + results), stream)
    assert second.finish == "stop" and "20" in second.message.content and not second.message.calls
    if stream:
        check_stream(chunks, second)


async def test_a_stream_cut_mid_answer_resets_its_step(recorded, llm):
    recorded("anthropic_cut_stream")  # anthropic_loop_stream's first reply, cut before message_stop
    s = await llm.query(LOOPS["anthropic"], ASK, stream=True)
    chunks = []
    with pytest.raises(ProviderError, match="ended before"):
        async for c in s:
            chunks.append(c)
    assert chunks[0] == {"type": "start-step"} and chunks[-1] == {"type": "reset-step"} and s.response is None


CONFIGURED = {  # each field placed per API; the recorded request body pins the spelling
    "anthropic": Model(provider="anthropic", model="claude-haiku-5-5", config=Config(
        system="Answer in French.", thinking="low", max_tokens=2000, cache=5, tools=[WEATHER],
        tool_choice=ToolChoice(mode="none"))),
    "openai": Model(provider="openai", model="gpt-5.6-luna", config=Config(
        system="Answer in French.", thinking="low", max_tokens=2000, cache=30, tools=[WEATHER],
        tool_choice=ToolChoice(mode="none"))),
    "gemini": Model(provider="gemini", model="gemini-3.5-flash", config=Config(
        system="Answer in French.", thinking="low", max_tokens=2000, cache="none", temperature=0.5, tools=[WEATHER],
        tool_choice=ToolChoice(mode="none"))),
    "openai-none": Model(provider="openai", model="gpt-5.6-luna", config=Config(system="Answer in French.",
                                                                                 cache="none")),
}


@pytest.mark.parametrize("provider", CONFIGURED)
async def test_config_fields_reach_the_request(provider, recorded, llm):
    recorded(f"{provider}_configured")
    r = await llm.query(CONFIGURED[provider], [{"role": "user", "content": "Say good morning to Toronto."}])
    assert r.finish == "stop" and "bonjour" in r.message.content.lower() and not r.message.calls


class Verdict(BaseModel):
    answer: int
    confidence: Literal["low", "high"]


VERDICT = {**Verdict.model_json_schema(), "additionalProperties": False}


@pytest.mark.parametrize("schema", [Verdict, VERDICT], ids=["class", "dict"])
@pytest.mark.parametrize("candidate", [Model(provider="anthropic", model="claude-haiku-5-5"),
                                       Model(provider="openai", model="gpt-5.6-luna"),
                                       Model(provider="gemini", model="gemini-3.5-flash")], ids=lambda m: m.provider)
async def test_parsed_holds_the_schema_object(candidate, schema, recorded, llm):
    recorded(f"{candidate.provider}_parsed" + ("" if schema is Verdict else "_dict"))
    candidate = candidate.model_copy(update={"config": Config(response_format=Format(type="schema", schema=schema))})
    r = await llm.query(candidate, [{"role": "user", "content": "What is 2+2? Give your confidence."}])
    parsed = r.parsed if schema is Verdict else Verdict.model_validate(r.parsed)
    assert isinstance(parsed, Verdict) and parsed.answer == 4


async def test_flagged_input_never_reaches_the_model(recorded, llm):
    recorded("moderation_flagged")  # one exchange: the moderation call; teardown fails on a second
    candidate = Model(provider="anthropic", model="claude-haiku-5-5", config=Config(moderation=True))
    with pytest.raises(Flagged):
        await llm.query(candidate, [{"role": "user", "content": "I will hunt you down and kill you."}])


async def test_a_call_that_cant_be_built_sends_nothing(recorded, llm):
    recorded("nothing_sent")  # no exchanges: teardown fails on any request
    with pytest.raises(UnsupportedField):
        await llm.query(Model(provider="gemini", model="gemini-3.5-flash", config=Config(cache=30)), ASK)


async def test_openai_audio_input_goes_to_completion(recorded, llm):
    recorded("openai_audio")  # the recorded path is /chat/completions: responses takes no audio
    spoken = Audio(str(FIXTURES / "spoken.mp3"))  # "The code is seven."
    r = await llm.query(Model(provider="openai", model="gpt-audio-mini"),
                        [{"role": "user", "content": [Text(text="Transcribe this audio."), spoken]}])
    assert "seven" in r.message.content.lower() or "7" in r.message.content


async def test_a_completion_stream_without_its_closing_usage_is_cut(recorded, llm):
    recorded("fireworks_stream_without_usage")  # fireworks_loop_stream's first reply, its usage chunk removed
    s = await llm.query(LOOPS["fireworks"], ASK, stream=True)
    chunks = []
    with pytest.raises(ProviderError, match="ended before"):
        async for c in s:
            chunks.append(c)
    assert chunks[-1] == {"type": "reset-step"} and s.response is None


# the streamed anthropic call is the sly one: eager_input_streaming sends the arguments unbuffered, and
# anthropic's SDK parses the cut JSON into a valid-looking input
CUT = {"openai": Model(provider="openai", model="gpt-5.6-luna",
                       config=Config(tools=[WRITE], thinking="none", max_tokens=16)),
       "anthropic": Model(provider="anthropic", model="claude-haiku-5-5", config=Config(
           tools=[{**WRITE, "function": {**WRITE["function"], "eager_input_streaming": True}}],
           tool_choice=ToolChoice(mode="required"), thinking="none", max_tokens=60))}


@pytest.mark.parametrize("provider, stream", [("openai", False), ("anthropic", True)])
async def test_a_tool_call_cut_by_max_tokens_raises_typed(provider, stream, recorded, llm):
    recorded(f"{provider}_cut_call" + ("_stream" if stream else ""))
    history = [{"role": "user", "content": "Call write_file with a 300-word poem as content."}]
    chunks = []
    with pytest.raises(CutCall):
        if stream:
            async for c in await llm.query(CUT[provider], history, stream=True):
                chunks.append(c)
        else:
            await llm.query(CUT[provider], history)
    if stream:
        assert chunks[-1] == {"type": "reset-step"} and "tool-input-available" not in {c["type"] for c in chunks}


class Essay(BaseModel):
    title: str
    body: str


class Rating(BaseModel):
    rating: int = Field(ge=1, le=5)  # anthropic can't enforce the range: it rides in the description


@pytest.mark.parametrize("name, schema, ask, finish", [
    ("anthropic_parsed_cut", Essay, "Write a 300-word essay about rivers.", "length"),
    ("anthropic_parsed_misfit", Rating, "Set rating to 9, exactly 9.", "stop"),  # its reply edited from 1 to 9
], ids=["cut", "misfit"])
async def test_an_answer_with_no_valid_object_keeps_its_text(name, schema, ask, finish, recorded, llm):
    recorded(name)
    candidate = Model(provider="anthropic", model="claude-haiku-5-5", config=Config(
        response_format=Format(type="schema", schema=schema), thinking="none", max_tokens=30))
    r = await llm.query(candidate, [{"role": "user", "content": ask}])
    assert r.finish == finish and r.parsed is None and r.message.content


@pytest.fixture
def held(provider_server):
    """anthropic_loop_stream's first event, then pings until the client hangs up; returns the hang-up."""
    hung_up = threading.Event()
    first = json.loads((FIXTURES / "anthropic_loop_stream.json").read_text())[0]["response"]["body"].split("\n\n")[0]

    def answer(handler, provider, path, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        try:
            handler.wfile.write(f"{first}\n\n".encode())
            for _ in range(500):  # 5 seconds at most
                time.sleep(0.01)
                handler.wfile.write(b'event: ping\ndata: {"type": "ping"}\n\n')
        except OSError:
            hung_up.set()
    provider_server(answer)
    return hung_up


@pytest.mark.parametrize("leave", ["aclose", "cancel"])
async def test_leaving_a_stream_closes_the_providers_stream(leave, held, llm):
    s = await llm.query(LOOPS["anthropic"], ASK, stream=True)
    if leave == "aclose":
        async for _ in s:
            break
        await s.aclose()
    else:  # cancelled inside the loop body, as while awaiting sse.send(chunk)
        forwarding = asyncio.Event()

        async def forward():
            async for _ in s:
                forwarding.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(forward())
        await asyncio.wait_for(forwarding.wait(), 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert await asyncio.to_thread(held.wait, 5) and s.response is None


# --- live: behaviour no recording can show ---

@pytest.mark.live
async def test_switching_models_mid_tool_loop(llm):
    history: list = list(ASK)
    for provider in ("gemini", "anthropic", "openai", "fireworks"):
        r = await llm.query(LOOPS[provider], history)
        history.append(r.message)
        for call in r.message.calls or []:
            history.append({"role": "tool", "call_id": call.id, "content": f"{call.arguments.get('city')}: 20C"})
        if not r.message.calls:
            history.append({"role": "user", "content": "Now check Rome too."})
    assert any(m.role == "tool" for m in validate(history)), "no model called the tool"


LONG = "Remember these facts. " + " ".join(f"Fact {i}: the code for item {i} is {i * 7919 % 10007}." for i in range(700))


@pytest.mark.live
@pytest.mark.parametrize("candidate", [
    Model(provider="anthropic", model="claude-haiku-5-5", config=Config(cache=5)),
    Model(provider="openai", model="gpt-5.6-luna"),
    Model(provider="openai", model="gpt-5.6-luna", config=Config(cache=30)),
    Model(provider="gemini", model="gemini-3.5-flash"),
    Model(provider="fireworks", model="accounts/fireworks/models/deepseek-v4p1-flash"),
], ids=["anthropic-5m", "openai-auto", "openai-30m", "gemini-auto", "fireworks-auto"])
async def test_a_saved_conversation_reads_the_cache(candidate, llm):
    history = [{"role": "user", "content": LONG + "\n\nWhat is the code for item 3?"}]
    first = await llm.query(candidate, history)
    later = saved(history + [first.message, {"role": "user", "content": "And for item 5?"}])
    second = await llm.query(candidate, later)
    assert second.usage.cache_read > 0, second.usage
