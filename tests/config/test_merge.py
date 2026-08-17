"""merge(): nested dicts deep-merge, scalars last-wins — extra is the collision customer."""
from llm_master.config import merge, thinking


def test_extra_deep_merges_into_reasoning():
    assert merge(thinking.translate("low", "responses"), {"reasoning": {"foo": 1}}) == {
        "reasoning": {"effort": "low", "summary": "auto", "foo": 1}}


def test_scalars_last_wins():
    assert merge({"temperature": 0.2}, {"temperature": 0.7}) == {"temperature": 0.7}


def test_lists_concat():
    assert merge({"tools": [{"a": 1}]}, {"tools": [{"b": 2}]}) == {"tools": [{"a": 1}, {"b": 2}]}
