import asyncio, base64, os
os.environ["ANY_LLM_UNIFIED_EXCEPTIONS"] = "1"
from PIL import Image, ImageDraw, ImageFont
from any_llm import AnyLLM

SCRATCH = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(SCRATCH, "uflo_test.pdf")
img = Image.new("RGB", (520, 160), "white")
d = ImageDraw.Draw(img)
d.text((30, 50), "UFLO PDF TEST CODE: ZEBRA-7741", fill="black", font=ImageFont.load_default(size=28))
img.save(pdf_path, "PDF")

uri = "data:application/pdf;base64," + base64.b64encode(open(pdf_path, "rb").read()).decode()
MSGS = [{"role": "user", "content": [
    {"type": "text", "text": "What is the code written in this document? Reply with only the code."},
    {"type": "file", "file": {"file_data": uri, "filename": "uflo_test.pdf"}},
]}]

async def main():
    for provider, model in [("anthropic", "claude-haiku-4-5"), ("openai", "gpt-5-nano"), ("gemini", "gemini-2.5-flash")]:
        try:
            llm = AnyLLM.create(provider)
            r = await llm.acompletion(model=model, messages=MSGS)
            u = r.usage
            print(f"{provider:10s} OK   answer={r.choices[0].message.content!r:34s} tokens in/out={u.prompt_tokens}/{u.completion_tokens}")
        except Exception as e:
            print(f"{provider:10s} FAIL {type(e).__name__}: {str(e)[:150]}")

asyncio.run(main())
