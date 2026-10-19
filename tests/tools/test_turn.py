"""The loop at its entry point, turn(), against recorded provider exchanges (RECORD=1 re-records them
live) and the Exa stand-in."""
import uuid
from contextlib import aclosing
from typing import Any

import pytest

from agent import Toolbox
from agent.sandbox import delete
from agent.turn import INTERRUPTED, STOPPED, Conversation, turn, typed_urls
from llm_master.types import Config, Model

MODEL = Model(provider="anthropic", model="claude-haiku-5-5", config=Config(system="Use the tools as the user asks."))
UNPINNED = ("tools",)  # test_toolbox keeps the schemas in step; a description's wording isn't the loop's
ASK = ("Use ask_user to ask me one question, which grade I teach, with the options Grade 9 and Grade 10. "
       "Then write one sentence for that grade.")


async def run(llm, conversation: Conversation, id: str = "c", **kwargs) -> list:
    return [chunk async for chunk in turn(llm, MODEL, id, conversation, **kwargs)]


PARALLEL = ("In one step, call todo with a two-step plan and web_search for a photosynthesis lesson. Then answer in "
            "one sentence, citing a result.")


async def test_parallel_calls_stream_their_results_and_the_loop_goes_on_to_the_answer(recorded, exa, llm):
    recorded("turn_parallel", UNPINNED)
    conversation = Conversation()
    chunks = await run(llm, conversation, text=PARALLEL)
    called = [c["toolName"] for c in chunks if c["type"] == "tool-input-available"]
    outputs = [c for c in chunks if c["type"] == "tool-output-available"]
    assert sorted(called) == ["todo", "web_search"] and len(outputs) == 2, chunks
    assert [c["url"] for c in chunks if c["type"] == "source-url"] == ["https://b.example", "https://c.example"]
    assert chunks[0] == {"type": "start"} and chunks[-1] == {"type": "finish"}
    assert "text-delta" in [c["type"] for c in chunks[chunks.index(outputs[-1]):]]  # the answer came after the results
    assert [m.role for m in conversation.messages] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [part["url"] for part in conversation.sources] == ["https://b.example", "https://c.example"]


async def test_a_closed_turn_answers_the_calls_it_cut_short(recorded, exa, llm):
    recorded("turn_closed", UNPINNED)
    conversation = Conversation()
    async with aclosing(turn(llm, MODEL, "c", conversation, text=PARALLEL)) as chunks:
        async for chunk in chunks:
            if isinstance(chunk, dict) and chunk["type"] == "tool-output-available":
                break  # the user stops the turn as the first result lands
    assert [m.content == STOPPED for m in conversation.messages if m.role == "tool"] == [False, True]


async def test_ask_user_ends_the_turn_and_its_answers_finish_it_next_request(recorded, llm):
    recorded("turn_ask", UNPINNED)
    conversation = Conversation()
    first = await run(llm, conversation, text=ASK)
    (call,) = conversation.messages[-1].calls or []
    assert call.name == "ask_user" and first[-1] == {"type": "finish"}
    assert not [c for c in first if c["type"].startswith("tool-output")]  # the browser asks; nothing answers yet
    questions: Any = call.arguments["questions"]
    question = questions[0]["question"]
    second = await run(llm, conversation, answers={call.id: {question: "Grade 9"}})
    assert f'"{question}"="Grade 9"' in str(conversation.messages[2].content)
    assert "9" in "".join(c["delta"] for c in second if c["type"] == "text-delta")


async def test_a_new_message_declines_the_waiting_question(recorded, llm):
    recorded("turn_decline", UNPINNED)
    conversation = Conversation()
    await run(llm, conversation, text=ASK)
    await run(llm, conversation, text="Never mind. Reply with the word ok.")
    declined, said = conversation.messages[2:4]
    assert (declined.role, declined.content, said.role) == ("tool", Toolbox("c").resume(None).content, "user")


async def test_a_stop_mid_answer_keeps_the_text_the_user_saw(recorded, llm):
    recorded("turn_interrupted", UNPINNED)
    conversation = Conversation()
    seen = ""
    async with aclosing(turn(llm, MODEL, "c", conversation, text="Count from 1 to 50, one number per line.")) as chunks:
        async for chunk in chunks:
            if isinstance(chunk, dict) and chunk["type"] == "text-delta":
                seen += chunk["delta"]
                break
    kept = [(m.role, m.content) for m in conversation.messages[1:]]
    assert seen and kept == [("assistant", seen), ("user", INTERRUPTED)], kept
    after = await run(llm, conversation, text="Go on.")
    answer = "".join(c["delta"] for c in after if c["type"] == "text-delta")
    assert answer.split()[0] == str(int(seen.split()[-1]) + 1), answer  # it counts on from the kept text


@pytest.mark.live  # on Modal, with the model's turn recorded
async def test_a_call_that_changes_the_sandbox_runs_before_the_calls_after_it(recorded, llm):
    recorded("turn_ordered", UNPINNED)
    conversation, id = Conversation(), f"test-{uuid.uuid4().hex[:12]}"
    try:
        await run(llm, conversation, id, text="Make two tool calls together in this one reply, before you see "
                                              "either result: bash with `echo fresh > note.txt`, and read on "
                                              "note.txt. Then say done.")
    finally:
        await delete(id)
    bash, read = conversation.messages[1].calls or []
    assert (bash.name, read.name) == ("bash", "read")
    assert [m.content for m in conversation.messages if m.call_id == read.id] == ["fresh"]


def test_typed_urls_leave_out_the_punctuation_around_them():
    text = ("See https://a.example/x. Or (https://en.wikipedia.org/wiki/Mole_(unit)), "
            "[notes](https://b.example/n?q=1), <https://c.example>! and [Https://d.example]")
    assert typed_urls(text) == ["https://a.example/x", "https://en.wikipedia.org/wiki/Mole_(unit)",
                                "https://b.example/n?q=1", "https://c.example", "Https://d.example"]
