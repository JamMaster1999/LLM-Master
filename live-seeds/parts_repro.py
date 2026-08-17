"""Live repro: nano-banana through acompletion on upstream main drops the image and the whole choice."""
import asyncio, os, pathlib

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
for line in pathlib.Path("/Users/sina/Desktop/Uflo Platform/uflo-AI-server/llm_master/.env").read_text().splitlines():
    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
        k, _, v = line.partition("=")
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

PROMPT = "Draw a simple red circle on a white background."
MODEL = "gemini-2.5-flash-image"

from any_llm import AnyLLM


async def main():
    llm = AnyLLM.create("gemini")
    resp = await llm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        response_modalities=["TEXT", "IMAGE"],
    )
    print("any-llm choices:", resp.choices)
    u = resp.usage
    print(f"any-llm usage:   prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")

asyncio.run(main())

from google import genai
from google.genai import types as g

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
raw = client.models.generate_content(
    model=MODEL, contents=PROMPT,
    config=g.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
)
print("raw parts:", [
    ("text" if p.text else f"inline_data {p.inline_data.mime_type}, {len(p.inline_data.data)} bytes")
    for p in raw.candidates[0].content.parts
])
