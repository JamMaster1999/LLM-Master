"""Evidence for two upstream issues. Prints usage numbers only — never prints keys."""
import asyncio, os, pathlib

for line in pathlib.Path("/Users/sina/Desktop/Uflo Platform/uflo-AI-server/llm_master/.env").read_text().splitlines():
    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
        k, _, v = line.partition("=")
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

from google import genai
from google.genai import types as g

MODEL = "gemini-2.5-flash"
PROMPT = "What is 15% of 240? Think it through carefully before answering."
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def show(tag, um):
    print(f"{tag}: prompt={um.prompt_token_count} candidates={um.candidates_token_count} "
          f"thoughts={um.thoughts_token_count} total={um.total_token_count}")


print("=== A: include_thoughts=False vs thinking_budget=0 (raw google-genai) ===")
r1 = client.models.generate_content(model=MODEL, contents=PROMPT,
    config=g.GenerateContentConfig(thinking_config=g.ThinkingConfig(include_thoughts=False)))
show("include_thoughts=False", r1.usage_metadata)
r2 = client.models.generate_content(model=MODEL, contents=PROMPT,
    config=g.GenerateContentConfig(thinking_config=g.ThinkingConfig(thinking_budget=0)))
show("thinking_budget=0    ", r2.usage_metadata)

print("\n=== B: any-llm usage vs its own total (main checkout) ===")
from any_llm import AnyLLM


async def b():
    llm = AnyLLM.create("gemini")
    resp = await llm.acompletion(model=MODEL, messages=[{"role": "user", "content": PROMPT}])
    u = resp.usage
    print(f"any-llm usage: prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
    print(f"prompt + completion = {u.prompt_tokens + u.completion_tokens}  (missing: {u.total_tokens - u.prompt_tokens - u.completion_tokens})")

asyncio.run(b())
