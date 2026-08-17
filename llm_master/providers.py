"""Rate-limit data: edit rows here when models change. Shapes live in types.py; SPEC section 1."""
from llm_master.types import ModelLimit, Provider

_GEMINI_MODELS = [
    ModelLimit(name="gemini-2.5-flash-lite", requests=30000),
    ModelLimit(name="gemini-2.5-flash-lite-preview-09-2025", requests=30000),
    ModelLimit(name="gemini-3.1-flash-lite", requests=30000),
    ModelLimit(name="gemini-3.1-flash-lite-preview", requests=30000),
    ModelLimit(name="gemini-3.5-flash", requests=20000),
    ModelLimit(name="gemini-3-flash-preview", requests=20000),
    ModelLimit(name="gemini-2.5-pro", requests=2000),
    ModelLimit(name="gemini-3-pro-preview", requests=2000),
    ModelLimit(name="gemini-3.1-pro-preview", requests=2000),
    ModelLimit(name="gemini-2.5-flash-image", requests=5000),
    ModelLimit(name="gemini-3-pro-image", requests=2000),
]

PROVIDERS = [
    Provider(
        name="openai",
        api="responses",
        search="tool",
        requests=10000,
        interval=60,
        models=[
            ModelLimit(name="gpt-5.1", requests=15000),
            ModelLimit(name="gpt-5.2", requests=15000),
            ModelLimit(name="gpt-5.4", requests=15000),
            ModelLimit(name="gpt-5.5", requests=15000),
            ModelLimit(name="gpt-5.6-sol", requests=15000),
            ModelLimit(name="gpt-5.6-terra", requests=15000),
            ModelLimit(name="gpt-5.6-luna", requests=15000),
            ModelLimit(name="chatgpt-4o-latest", requests=5000),
            ModelLimit(name="gpt-audio-mini", requests=5000),
            ModelLimit(name="gpt-4o-audio-preview", requests=5000),
            ModelLimit(name="omni-moderation-latest", requests=1000),
            ModelLimit(name="gpt-image-1", requests=50),
        ],
    ),
    Provider(name="anthropic", api="messages", search="tool", requests=4000, interval=60, models=[]),
    Provider(name="gemini", api="completion", search="tool", price_id="google", requests=10000, interval=60, models=_GEMINI_MODELS),
    Provider(name="vertexai", api="completion", search="tool", price_id="google", requests=10000, interval=60, models=_GEMINI_MODELS),
    Provider(name="fireworks", api="completion", search="none", requests=600, interval=60, models=[]),
]

_BY_NAME = {p.name: p for p in PROVIDERS}
assert len(_BY_NAME) == len(PROVIDERS), "duplicate provider names"


def limit(provider: str, model: str) -> tuple[int, int]:
    """(requests, interval) for one bucket. A model override replaces the provider default."""
    p = _BY_NAME[provider]  # unknown provider fails loud, on purpose
    m = next((x for x in p.models if x.name == model), None)
    return (
        m.requests if m and m.requests else p.requests,
        m.interval if m and m.interval else p.interval,
    )
