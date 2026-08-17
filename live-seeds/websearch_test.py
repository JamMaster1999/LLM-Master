"""Live test: server-side web search through any-llm on anthropic, openai, gemini.
Prints structure only — never prints keys."""
import asyncio, os

os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
from any_llm import AnyLLM  # noqa: E402

Q = "What is the most recent stable version of Python, as of this week? One short sentence."


def clip(s, n=140):
    s = (s or "").replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


async def test_anthropic():
    print("\n========== ANTHROPIC amessages() + web_search ==========")
    llm = AnyLLM.create("anthropic")
    resp = None
    for model in ["claude-haiku-4-5", "claude-3-5-haiku-latest"]:
        try:
            resp = await llm.amessages(
                model=model, max_tokens=800,
                messages=[{"role": "user", "content": Q}],
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            )
            print(f"model: {model}")
            break
        except Exception as e:
            print(f"{model}: {type(e).__name__}: {clip(str(e))}")
    if resp is None:
        return
    print("content block types, in order:", [b.type for b in resp.content])
    n_cit = 0
    first_cit = None
    for b in resp.content:
        if b.type == "text":
            for c in (getattr(b, "citations", None) or []):
                n_cit += 1
                if first_cit is None:
                    first_cit = f"{getattr(c, 'title', '?')} | {getattr(c, 'url', '?')}"
    print(f"citations on text blocks: {n_cit}; first: {clip(first_cit or 'NONE')}")
    stu = getattr(resp.usage, "server_tool_use", None)
    print("usage.server_tool_use:", stu)
    text = "".join(b.text for b in resp.content if b.type == "text")
    print("answer:", clip(text))


async def test_openai():
    print("\n========== OPENAI aresponses() + web_search ==========")
    llm = AnyLLM.create("openai")
    resp = None
    for model in ["gpt-5-nano", "gpt-5-mini", "gpt-4.1-mini"]:
        try:
            resp = await llm.aresponses(model, Q, tools=[{"type": "web_search"}])
            print(f"model: {model}")
            break
        except Exception as e:
            print(f"{model}: {type(e).__name__}: {clip(str(e))}")
    if resp is None:
        return
    print("output item types, in order:", [it.type for it in resp.output])
    for it in resp.output:
        if it.type == "web_search_call":
            action = getattr(it, "action", None)
            print(f"  web_search_call: status={getattr(it, 'status', '?')} action={clip(str(action), 100)}")
    n_ann = 0
    first_ann = None
    answer = ""
    for it in resp.output:
        if it.type == "message":
            for part in it.content:
                answer += getattr(part, "text", "") or ""
                for a in (getattr(part, "annotations", None) or []):
                    n_ann += 1
                    if first_ann is None:
                        first_ann = f"{getattr(a, 'type', '?')} | {getattr(a, 'title', '?')} | {getattr(a, 'url', '?')}"
    print(f"citations (annotations): {n_ann}; first: {clip(first_ann or 'NONE')}")
    print("answer:", clip(answer))


async def test_gemini():
    print("\n========== GEMINI acompletion() + google_search (via any-llm) ==========")
    from google.genai import types as g
    llm = AnyLLM.create("gemini")
    resp = None
    used = None
    for model in ["gemini-2.5-flash", "gemini-3.0-flash", "gemini-flash-latest"]:
        try:
            resp = await llm.acompletion(
                model=model, messages=[{"role": "user", "content": Q}],
                tools=[g.Tool(google_search=g.GoogleSearch())],
            )
            used = model
            print(f"model: {model}")
            break
        except Exception as e:
            print(f"{model}: {type(e).__name__}: {clip(str(e))}")
    if resp is None:
        return
    msg = resp.choices[0].message
    print("message.content:", clip(msg.content))
    print("message.tool_calls:", msg.tool_calls)
    extra = getattr(msg, "model_extra", None) or {}
    print("extra fields on message:", list(extra.keys()) or "NONE")
    r_extra = (getattr(resp, "model_extra", None) or {})
    print("extra fields on response:", list(r_extra.keys()) or "NONE")

    print("\n---------- same call, RAW google-genai client ----------")
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    raw = client.models.generate_content(
        model=used, contents=Q,
        config=g.GenerateContentConfig(tools=[g.Tool(google_search=g.GoogleSearch())]),
    )
    cand = raw.candidates[0]
    gm = cand.grounding_metadata
    if gm is None:
        print("grounding_metadata: NONE")
    else:
        queries = getattr(gm, "web_search_queries", None)
        chunks = getattr(gm, "grounding_chunks", None) or []
        supports = getattr(gm, "grounding_supports", None) or []
        print(f"grounding_metadata: queries={queries}")
        print(f"  grounding_chunks: {len(chunks)}; first: "
              + clip(f"{chunks[0].web.title} | {chunks[0].web.uri}" if chunks else "NONE"))
        print(f"  grounding_supports (citation spans): {len(supports)}")
    print("raw answer:", clip(raw.text))


async def main():
    for t in (test_anthropic, test_openai, test_gemini):
        try:
            await t()
        except Exception as e:
            print(f"FAILED {t.__name__}: {type(e).__name__}: {clip(str(e), 200)}")


asyncio.run(main())
