"""Same-response repro: raw gemini usage_metadata vs any-llm's converted usage."""
import os, pathlib

for line in pathlib.Path("/Users/sina/Desktop/Uflo Platform/uflo-AI-server/llm_master/.env").read_text().splitlines():
    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
        k, _, v = line.partition("=")
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

from google import genai
from any_llm.providers.gemini.utils import _extract_usage_dict

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 15% of 240? Think it through carefully before answering.",
)
m = resp.usage_metadata
print(f"raw:       prompt_token_count={m.prompt_token_count}  candidates_token_count={m.candidates_token_count}  "
      f"thoughts_token_count={m.thoughts_token_count}  total_token_count={m.total_token_count}")
print(f"converted: {_extract_usage_dict(resp)}")
