"""Live: reasoning_effort='none' on the fix branch. Gap = thoughts billed (this branch has no usage fix)."""
import asyncio, os, pathlib

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
for line in pathlib.Path("/Users/sina/Desktop/Uflo Platform/uflo-AI-server/llm_master/.env").read_text().splitlines():
    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
        k, _, v = line.partition("=")
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

from any_llm import AnyLLM

Q = "What is 15% of 240? Think it through carefully before answering."


async def main():
    llm = AnyLLM.create("gemini")
    resp = await llm.acompletion(model="gemini-2.5-flash", messages=[{"role": "user", "content": Q}],
                                 reasoning_effort="none")
    u = resp.usage
    print(f"flash 'none': prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens} "
          f"hidden-thought gap={u.total_tokens - u.prompt_tokens - u.completion_tokens}")

    try:
        await llm.acompletion(model="gemini-2.5-pro", messages=[{"role": "user", "content": Q}],
                              reasoning_effort="none")
        print("pro 'none': unexpectedly succeeded")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:220]
        print(f"pro 'none': {type(e).__name__}: {msg}")

asyncio.run(main())
