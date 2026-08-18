"""Live-test env: keys from the repo .env, typed any-llm errors on."""
import os
import pathlib

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
_ENV = pathlib.Path(__file__).parent.parent / "llm_master" / ".env"
if _ENV.exists():  # absent on fresh checkouts: offline tests need no keys
    for _line in _ENV.read_text().splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep:
            os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))
