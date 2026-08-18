"""Reads of the provider page. The page's value rules are import-time asserts in registry.py."""
import pytest

from llm_master.registry import limit, search


def test_model_override_replaces_provider_default():
    assert limit("openai", "gpt-5.6-sol") == (15000, 60)


def test_unknown_model_gets_provider_default():
    assert limit("openai", "brand-new-model") == (10000, 60)


def test_provider_with_no_overrides():
    assert limit("anthropic", "claude-sonnet-5") == (4000, 60)


def test_unknown_provider_raises():
    with pytest.raises(KeyError):
        limit("closedai", "gpt-5.1")


def test_search_accessor_reads_the_row():
    assert search("gemini") == "tool"
    assert search("fireworks") == "none"
    with pytest.raises(KeyError):
        search("nope")
