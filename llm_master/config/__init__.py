"""Bucket 1: one module per config field — the field's type and its per-API translation."""
from typing import Literal

API = Literal["completion", "messages", "responses"]


def merge(*fragments: dict) -> dict:
    """Combine translation fragments into one request: nested dicts deep-merge (extra into
    reasoning), lists concatenate (web_search's tool joins the caller's tools), scalars
    last-wins. Candidate extra merges last."""
    out: dict = {}
    for fragment in fragments:
        for key, value in fragment.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value)
            elif isinstance(value, list) and isinstance(out.get(key), list):
                out[key] = out[key] + value
            else:
                out[key] = value
    return out


class UnsupportedField(Exception):
    """A config field has no translation on this API. Fallback moves on."""

    def __init__(self, api: str, field: str, why: str):
        self.api, self.field = api, field
        super().__init__(f"{api}: {field} — {why}")
