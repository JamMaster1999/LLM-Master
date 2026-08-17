"""Value rules for providers.py — the pre-commit checks from SPEC section 1. No network."""
import pytest
from pydantic import ValidationError

from llm_master.providers import limit
from llm_master.types import ModelLimit, Provider


def test_model_override_replaces_provider_default():
    assert limit("openai", "gpt-5.6-sol") == (15000, 60)


def test_unknown_model_gets_provider_default():
    assert limit("openai", "brand-new-model") == (10000, 60)


def test_provider_with_no_overrides():
    assert limit("anthropic", "claude-sonnet-5") == (4000, 60)


def test_unknown_provider_raises():
    with pytest.raises(KeyError):
        limit("closedai", "gpt-5.1")


def test_override_must_override_something():
    with pytest.raises(ValidationError):
        ModelLimit(name="pointless")


def test_duplicate_models_rejected():
    with pytest.raises(ValidationError):
        Provider(name="p", api="completion", search="none", requests=1, interval=60,
                 models=[ModelLimit(name="m", requests=1), ModelLimit(name="m", requests=2)])


def test_api_and_search_are_required():
    with pytest.raises(ValidationError):
        Provider(name="p", search="none", requests=1, interval=60, models=[])  # no api
    with pytest.raises(ValidationError):
        Provider(name="p", api="completion", requests=1, interval=60, models=[])  # no search
