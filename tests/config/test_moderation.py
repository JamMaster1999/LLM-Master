"""moderation pre-check, live both directions — the endpoint is free."""
import pytest

from llm_master.config.moderation import Flagged, check
from llm_master.io.messages import validate

pytestmark = pytest.mark.live


async def test_benign_input_passes(llm):
    await check(llm, validate([{"role": "user", "content": "What's the weather in Toronto?"}]))


async def test_violent_input_flagged_with_categories(llm):
    with pytest.raises(Flagged) as e:
        await check(llm, validate([{"role": "user", "content": "I will hunt you down and kill you."}]))
    assert e.value.categories, "flagged input must name its categories"
