"""Live check: parallel_tool_calls alone must not force a tool call."""
import asyncio, os, pathlib

for line in pathlib.Path("/Users/sina/Desktop/Uflo Platform/uflo-AI-server/llm_master/.env").read_text().splitlines():
    if line.startswith("ANTHROPIC_API_KEY="):
        k, _, v = line.partition("=")
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

from any_llm import AnyLLM

TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


async def main():
    llm = AnyLLM.create("anthropic")
    resp = await llm.acompletion(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "Just say the word 'hello'. Do not use any tools."}],
        tools=[TOOL],
        parallel_tool_calls=True,
    )
    msg = resp.choices[0].message
    print("tool_calls:", msg.tool_calls)
    print("content:", (msg.content or "")[:80])
    print("PASS: model was free to answer in text" if not msg.tool_calls else "FAIL: tool call was forced")

asyncio.run(main())
