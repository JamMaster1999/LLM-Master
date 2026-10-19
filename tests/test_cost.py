"""Cost: each call's usage in dollars, every part at its own rate. See SPEC.md Money.

The expected dollars come from genai-prices itself, fed the token counts each recording reports.
"""
import genai_prices
import pytest
from test_media import DRAW, GEMINI, OPENAI, parent
from test_query import FIXTURES, LONG

from llm_master.io.parts import Audio, Text
from llm_master.types import Config, Model


async def gemini_image(llm):
    return await llm.image(GEMINI, DRAW)


async def openai_edit(llm):
    return await llm.image(OPENAI, "Make the square blue.", [parent()])


async def openai_audio(llm):
    spoken = Audio(str(FIXTURES / "spoken.mp3"))
    return await llm.query(Model(provider="openai", model="gpt-audio-mini"),
                           [{"role": "user", "content": [Text(text="Transcribe this audio."), spoken]}])


async def anthropic_cache_1h(llm):  # streamed: a stream's usage is priced like a plain call's
    candidate = Model(provider="anthropic", model="claude-haiku-5-5", config=Config(max_tokens=50),
                      extra={"cache_control": {"ttl": "1h"}})
    stream = await llm.query(candidate, [{"role": "user", "content": LONG + "\n\nWhat is the code for item 3?"}],
                             stream=True)
    async for _ in stream:
        pass
    return stream.response


@pytest.mark.parametrize("call, provider, model, tokens", [
    (gemini_image, "google", "gemini-3.1-flash-image",
     {"input_tokens": 12, "output_tokens": 1086, "output_image_tokens": 747}),
    (openai_edit, "openai", "gpt-image-2",
     {"input_tokens": 27, "input_image_tokens": 16, "output_tokens": 196, "output_image_tokens": 196}),
    (openai_audio, "openai", "gpt-audio-mini", {"input_tokens": 32, "input_audio_tokens": 16, "output_tokens": 6}),
    (anthropic_cache_1h, "anthropic", "claude-haiku-5-5",
     {"input_tokens": 12555, "cache_write_tokens": 12551, "cache_write_1h_tokens": 12551, "output_tokens": 23}),
], ids=lambda value: getattr(value, "__name__", None))
async def test_each_part_is_priced_at_its_own_rate(call, provider, model, tokens, recorded, llm):
    recorded(call.__name__)
    made = await call(llm)
    expected = genai_prices.calc_price(genai_prices.Usage(**tokens), model, provider_id=provider).total_price
    assert made.cost == pytest.approx(float(expected))


async def test_unknown_usage_costs_none(recorded, llm):
    recorded("openai_speech_plain")  # the plain format reports no usage
    spoken = await llm.speech(Model(provider="openai", model="gpt-4o-mini-tts", extra={"stream_format": "audio"}),
                              "Say: hello there.")
    assert spoken.cost is None


async def test_a_model_genai_prices_lacks_costs_none(recorded, llm):
    recorded("fireworks_unpriced")  # glm-5p3's reply, renamed to a model no price list will ever hold
    r = await llm.query(Model(provider="fireworks", model="accounts/fireworks/models/unpriced"),
                        [{"role": "user", "content": "Say hi."}])
    assert r.usage and r.cost is None
