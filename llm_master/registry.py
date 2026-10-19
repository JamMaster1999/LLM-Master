"""Reads of the providers page, plus the import-time checks that keep it honest."""
from typing import get_args

from genai_prices.types import ProviderID

from llm_master.providers import API, PROVIDERS, SDK, ProviderName
from llm_master.shapes import ProviderRow


def price_id(provider: ProviderName) -> str:
    """The provider's name in genai-prices."""
    return PROVIDERS[provider].get("price_id", provider)


assert set(PROVIDERS) == set(get_args(ProviderName)), "ProviderName and PROVIDERS keys disagree"
for _name, _row in PROVIDERS.items():
    assert _row.keys() >= ProviderRow.__required_keys__, f"{_name}: missing {ProviderRow.__required_keys__ - _row.keys()}"
    assert _row["api"] in get_args(API), f"{_name}: unknown api {_row['api']!r}"
    assert _row.get("audio_api", _row["api"]) in get_args(API), f"{_name}: unknown audio_api"
    assert _row["sdk"] in get_args(SDK), f"{_name}: unknown sdk {_row['sdk']!r}"
    assert price_id(_name) in get_args(ProviderID), f"{_name}: genai-prices knows no {price_id(_name)!r}"
