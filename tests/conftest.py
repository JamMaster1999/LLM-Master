"""Live-test env: keys from the v1 .env, typed any-llm errors on."""
import os
import pathlib

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
_ENV = pathlib.Path("/Users/sina/Desktop/Uflo Platform/uflo-AI-server/llm_master/.env")
for _line in _ENV.read_text().splitlines():
    _key, _sep, _value = _line.partition("=")
    if _sep:
        os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))
