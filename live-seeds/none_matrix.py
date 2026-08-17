"""Comprehensive live matrix: reasoning_effort='none' across every code path on the fix branch."""
import asyncio, logging, os, pathlib

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
for line in pathlib.Path("/Users/sina/Desktop/Uflo Platform/uflo-AI-server/llm_master/.env").read_text().splitlines():
    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
        k, _, v = line.partition("=")
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError

records: list[logging.LogRecord] = []


class Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        records.append(record)


logging.getLogger().addHandler(Capture())
logging.getLogger().setLevel(logging.WARNING)
for name in list(logging.Logger.manager.loggerDict):
    if "any_llm" in name:
        logging.getLogger(name).setLevel(logging.WARNING)

Q = "What is 15% of 240? Think it through carefully before answering."

MODELS = [
    ("gemini-2.5-flash",       "budget, disableable"),
    ("gemini-2.5-flash-lite",  "budget, disableable"),
    ("gemini-2.5-pro",         "budget, thinking-only"),
    ("gemini-3-flash-preview", "budget (3.0)"),
    ("gemini-3.1-flash-lite",  "budget (3.1)"),
    ("gemini-3.1-pro-preview", "budget (3.1 pro)"),
    ("gemini-3.5-flash",       "level -> clamp MINIMAL"),
    ("gemini-3.5-flash-lite",  "level -> clamp MINIMAL"),
    ("gemini-3.6-flash",       "level -> clamp MINIMAL"),
    ("gemini-3.7-flash",       "level -> clamp MINIMAL"),
    ("gemini-9-flash",         "nonexistent"),
]


async def main() -> None:
    llm = AnyLLM.create("gemini")
    for model, note in MODELS:
        records.clear()
        try:
            resp = await llm.acompletion(model=model, messages=[{"role": "user", "content": Q}],
                                         reasoning_effort="none", max_tokens=2000)
            u = resp.usage
            gap = u.total_tokens - u.prompt_tokens - u.completion_tokens
            warned = any("cannot disable thinking" in r.getMessage() for r in records)
            print(f"OK    {model:26s} [{note}] gap={gap} warn={warned} "
                  f"(p={u.prompt_tokens} c={u.completion_tokens} t={u.total_tokens})")
        except Exception as e:
            typed = isinstance(e, AnyLLMError)
            warned = any("cannot disable thinking" in r.getMessage() for r in records)
            msg = str(e).replace("\n", " ")[:110]
            print(f"ERR   {model:26s} [{note}] {type(e).__name__} typed={typed} warn={warned} | {msg}")

asyncio.run(main())
