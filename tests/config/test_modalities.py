"""modalities gate: text passes, everything else raises typed on every API."""
from typing import get_args

import pytest

from llm_master.config import UnsupportedField, modalities
from llm_master.providers import API

APIS = get_args(API)


def test_text_only_says_nothing():
    for api in APIS:
        assert modalities.translate(["text"], api) == {}
        assert modalities.translate([], api) == {}


def test_non_text_raises_typed_everywhere():
    for api in APIS:
        for asked in (["image"], ["audio"], ["text", "image"]):
            with pytest.raises(UnsupportedField) as e:
                modalities.translate(asked, api)
            assert e.value.field == "modalities" and e.value.api == api
