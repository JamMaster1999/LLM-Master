"""Usage to dollars, through genai-prices. See SPEC.md Money."""
import genai_prices

from llm_master.io.response import Usage
from llm_master.providers import ProviderName
from llm_master.registry import price_id


def price(usage: Usage | None, provider: ProviderName, model: str) -> float | None:
    """One call's dollars; None when its usage or the model's price is unknown, never a guessed zero."""
    if usage is None:
        return None
    tokens = genai_prices.Usage(
        input_tokens=usage.input, output_tokens=usage.output, cache_read_tokens=usage.cache_read,
        cache_write_tokens=usage.cache_write, cache_write_1h_tokens=usage.cache_write_1h,
        input_audio_tokens=usage.input_audio, input_image_tokens=usage.input_image,
        output_audio_tokens=usage.output_audio, output_image_tokens=usage.output_image)
    try:
        return float(genai_prices.calc_price(tokens, model, provider_id=price_id(provider)).total_price)
    # a model genai-prices doesn't know yet (registry.py checks the provider at import), or usage its price
    # entry can't split, such as cached audio on a model with no such rate
    except (LookupError, ValueError):
        return None
