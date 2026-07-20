"""
Modal-based latency comparison for multiple LLM providers.

This benchmark measures end-to-end latency for:
- Native clients (imports + initialization + first token)
- QueryLLM (import + initialization + first token)

IMPORTANT: All native clients run *before* importing QueryLLM so we can
measure the true cold-start import cost for each provider individually.
"""

import modal
import os

app = modal.App(name="llm_latency_benchmark_all_providers")

benchmark_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("nest_asyncio", "requests")
    .add_local_dir(".", remote_path="/root/llm_master")
)


@app.function(
    image=benchmark_image,
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("posthog-public-project"),
    ],
    timeout=900,
    cpu=4.0,
    memory=12000,
)
async def run_all_providers_benchmark() -> None:
    """Run native benchmarks first, then QueryLLM benchmarks."""
    import sys
    import time

    sys.path.insert(0, "/root/llm_master")
    os.environ["LLM_RATE_LIMIT_MODE"] = "modal"

    prompt = "Write a two sentence story about a cat and a dog"
    messages = [{"role": "user", "content": prompt}]

    print("=" * 80)
    print("LLM PROVIDER LATENCY BENCHMARK (END-TO-END)")
    print("=" * 80)
    print("\nStage 1: Native client imports + init + first token (cold start)")
    print("Stage 2: QueryLLM import + init + first token (after clearing caches)\n")

    native_results = {
        "anthropic": measure_native_anthropic(messages),
        "gemini": measure_native_gemini(messages, prompt),
        "openai_responses": measure_native_openai_responses(messages),
        "openai_chat": measure_native_openai_chat(messages),
    }

    clear_provider_modules()

    print("=" * 80)
    print("Native benchmarks complete. Provider modules cleared from cache.")
    print("Importing QueryLLM will now reflect a true cold start.")
    print("=" * 80)

    query_import_start = time.time()
    from llm_master import QueryLLM, LLMConfig  # type: ignore

    query_import_time = time.time() - query_import_start
    query_init_start = time.time()
    config = LLMConfig.from_env()
    llm = QueryLLM(config)
    query_init_time = time.time() - query_init_start

    print(f"QueryLLM import time: {query_import_time * 1000:.2f} ms")
    print(f"QueryLLM init time:   {query_init_time * 1000:.2f} ms\n")

    query_results = {
        "anthropic": await measure_query_llm(
            llm,
            model_name="claude-sonnet-4-5-20250929",
            messages=messages,
        ),
        "gemini": await measure_query_llm(
            llm,
            model_name="googleai:gemini-2.5-flash",
            messages=messages,
            reasoning={"thinking_budget": 0},
        ),
        "openai_responses": await measure_query_llm(
            llm,
            model_name="responses-gpt-4.1",
            messages=messages,
        ),
        "openai_chat": await measure_query_llm(
            llm,
            model_name="gpt-4.1",
            messages=messages,
        ),
    }

    print_summary(native_results, query_results, query_import_time, query_init_time)


def clear_provider_modules() -> None:
    """Remove provider-related modules from sys.modules to force cold imports."""
    import sys

    prefixes = (
        "anthropic",
        "google",
        "openai",
        "posthog",
    )
    modules_to_clear = []
    for name in list(sys.modules.keys()):
        for prefix in prefixes:
            if name == prefix or name.startswith(f"{prefix}."):
                modules_to_clear.append(name)
                break

    for name in modules_to_clear:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Native benchmarks
# ---------------------------------------------------------------------------

def measure_native_anthropic(messages):
    import time
    result = {}
    try:
        total_start = time.time()
        from anthropic import Anthropic
        result["import_time"] = time.time() - total_start

        init_start = time.time()
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        result["init_time"] = time.time() - init_start

        api_start = time.time()
        first_chunk = None
        chunk_count = 0
        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                if first_chunk is None:
                    first_chunk = time.time()
                chunk_count += 1

        result["first_token"] = (first_chunk - api_start) if first_chunk else None
        result["total"] = (first_chunk - total_start) if first_chunk else None
        result["chunks"] = chunk_count
        print_native_result("Anthropic", result)
    except Exception as exc:  # pragma: no cover - best effort logging
        result["error"] = str(exc)
        print(f"Native Anthropic test failed: {exc}\n")
    return result


def measure_native_gemini(messages, prompt):
    import time
    result = {}
    try:
        total_start = time.time()
        from google import genai
        from google.genai import types
        result["import_time"] = time.time() - total_start

        init_start = time.time()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        result["init_time"] = time.time() - init_start

        api_start = time.time()
        first_chunk = None
        chunk_count = 0
        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
            contents=[prompt],
        )
        for chunk in stream:
            if first_chunk is None:
                first_chunk = time.time()
            chunk_count += 1

        result["first_token"] = (first_chunk - api_start) if first_chunk else None
        result["total"] = (first_chunk - total_start) if first_chunk else None
        result["chunks"] = chunk_count
        print_native_result("Gemini", result)
    except Exception as exc:
        result["error"] = str(exc)
        print(f"Native Gemini test failed: {exc}\n")
    return result


def measure_native_openai_responses(messages):
    import time
    result = {}
    try:
        total_start = time.time()
        from openai import OpenAI
        result["import_time"] = time.time() - total_start

        init_start = time.time()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result["init_time"] = time.time() - init_start

        api_start = time.time()
        first_chunk = None
        chunk_count = 0
        stream = client.responses.create(
            model="gpt-4.1",
            input=messages,
            stream=True,
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                if first_chunk is None:
                    first_chunk = time.time()
                chunk_count += 1

        result["first_token"] = (first_chunk - api_start) if first_chunk else None
        result["total"] = (first_chunk - total_start) if first_chunk else None
        result["chunks"] = chunk_count
        print_native_result("OpenAI Responses", result)
    except Exception as exc:
        result["error"] = str(exc)
        print(f"Native OpenAI Responses test failed: {exc}\n")
    return result


def measure_native_openai_chat(messages):
    import time
    result = {}
    try:
        total_start = time.time()
        from openai import OpenAI
        result["import_time"] = time.time() - total_start

        init_start = time.time()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result["init_time"] = time.time() - init_start

        api_start = time.time()
        first_chunk = None
        chunk_count = 0
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                if first_chunk is None:
                    first_chunk = time.time()
                chunk_count += 1

        result["first_token"] = (first_chunk - api_start) if first_chunk else None
        result["total"] = (first_chunk - total_start) if first_chunk else None
        result["chunks"] = chunk_count
        print_native_result("OpenAI Chat", result)
    except Exception as exc:
        result["error"] = str(exc)
        print(f"Native OpenAI Chat test failed: {exc}\n")
    return result


def print_native_result(name: str, result: dict) -> None:
    if "error" in result:
        return
    print(f"[Native] {name}")
    print(f"  Import time: {result['import_time'] * 1000:.2f} ms")
    print(f"  Init time:   {result['init_time'] * 1000:.2f} ms")
    print(f"  First token: {result['first_token'] * 1000:.2f} ms")
    print(f"  TOTAL:       {result['total'] * 1000:.2f} ms")
    print(f"  Chunks:      {result['chunks']}\n")


# ---------------------------------------------------------------------------
# QueryLLM benchmarks
# ---------------------------------------------------------------------------

async def measure_query_llm(llm, *, model_name: str, messages, **kwargs) -> dict:
    import time

    result: dict = {"model": model_name}
    try:
        api_start = time.time()
        first_chunk = None
        chunk_count = 0

        stream = await llm.query(
            model_name=model_name,
            messages=messages,
            stream=True,
            moderation=False,
            **kwargs,
        )

        async for chunk in stream:
            if first_chunk is None:
                first_chunk = time.time()
            chunk_count += 1

        result["first_token"] = (first_chunk - api_start) if first_chunk else None
        result["chunks"] = chunk_count
    except Exception as exc:
        result["error"] = str(exc)
        print(f"QueryLLM test failed for {model_name}: {exc}\n")
        return result

    print(f"[QueryLLM] {model_name}")
    print(f"  First token: {result['first_token'] * 1000:.2f} ms")
    print(f"  Chunks:      {result['chunks']}\n")
    return result


def print_summary(native_results, query_results, import_time, init_time) -> None:
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"QueryLLM aggregate import (providers+deps): {import_time * 1000:.2f} ms")
    print(f"QueryLLM init time:                       {init_time * 1000:.2f} ms\n")

    names = {
        "anthropic": "Anthropic (Sonnet 4.5)",
        "gemini": "Google Gemini 2.5 Flash",
        "openai_responses": "OpenAI GPT-4.1 (Responses)",
        "openai_chat": "OpenAI GPT-4.1 (Chat)",
    }

    header = (
        f"{'Provider':35} | {'Native Import (ms)':>16} | {'Native TTFT (ms)':>16} | "
        f"{'Query TTFT (ms)':>15} | {'Δ TTFT (ms)':>12}"
    )
    print(header)
    print("-" * len(header))

    for key, label in names.items():
        native = native_results.get(key, {})
        query = query_results.get(key, {})

        native_import = native.get("import_time")
        native_ttft = native.get("first_token")
        query_ttft = query.get("first_token")

        if None in (native_import, native_ttft, query_ttft):
            line = f"{label:35} | {'n/a':>16} | {'n/a':>16} | {'n/a':>15} | {'n/a':>12}"
        else:
            delta = query_ttft - native_ttft
            line = (
                f"{label:35} | {native_import * 1000:16.2f} | {native_ttft * 1000:16.2f} | "
                f"{query_ttft * 1000:15.2f} | {delta * 1000:12.2f}"
            )
        print(line)

    print("\nTTFT = Time to first token. QueryLLM figures exclude import/init overhead")
    print("(reported separately above) so you can allocate those costs as needed.")
    print("Native columns reflect true cold-start imports because provider modules")
    print("were unloaded before QueryLLM import.\n")


@app.local_entrypoint()
def main() -> None:
    run_all_providers_benchmark.remote()
