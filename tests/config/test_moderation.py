"""moderation pre-check, live both directions — the endpoint is free."""
import pytest
from any_llm import AnyLLM

from llm_master.config.moderation import Flagged, check
from llm_master.io.messages import validate

pytestmark = pytest.mark.live


async def test_benign_input_passes():
    await check(AnyLLM.create("openai"), validate([{"role": "user", "content": "What's the weather in Toronto?"}]))


async def test_violent_input_flagged_with_categories():
    with pytest.raises(Flagged) as e:
        await check(AnyLLM.create("openai"),
                    validate([{"role": "user", "content": "I will hunt you down and kill you."}]))
    assert e.value.categories, "flagged input must name its categories"
