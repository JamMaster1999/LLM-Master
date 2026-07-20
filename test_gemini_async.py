import argparse
import asyncio
import os
import sys
import time
sys.path.insert(0, "/Users/sina/Desktop/Uflo Platform/gradescope-v2/server")

from dotenv import load_dotenv
load_dotenv("/Users/sina/Desktop/Uflo Platform/gradescope-v2/server/.env")

from llm_master_repo.response_synthesizer import QueryLLM
from google import genai

MODEL = "vertexai:gemini-3-flash-preview"
NATIVE_MODEL = "gemini-3-flash-preview"

def progress(done, total):
    bar_len = 40
    filled = int(bar_len * done / total) if total else bar_len
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {done}/{total}", end="", flush=True)
    if done == total:
        print()

async def main(num_requests, num_rounds):
    llm = QueryLLM()
    requests = [
        {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "kwargs": {"reasoning_effort": "none"},
        }
    ] * num_requests

    total_responses = 0
    total_start = time.perf_counter()

    for r in range(1, num_rounds + 1):
        print(f"\n── Round {r}/{num_rounds} ({num_requests} requests) ──")
        start = time.perf_counter()

        try:
            responses = await asyncio.wait_for(
                llm.async_query(model_name=MODEL, requests=requests, progress_callback=progress),
                timeout=60,
            )
        except asyncio.TimeoutError:
            print(f"\n  ERROR: Round {r} timed out after 60s — requests still hanging!")
            sys.exit(1)

        elapsed = time.perf_counter() - start
        total_responses += len(responses)
        print(f"\n  {len(responses)} responses in {elapsed:.1f}s ({len(responses)/elapsed:.1f} req/s)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*50}")
    print(f"Total: {total_responses} responses in {total_elapsed:.1f}s ({total_responses/total_elapsed:.1f} req/s)")

async def main_native(num_requests, num_rounds):
    """Same test using the Google GenAI SDK directly (no llm_master_repo)."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY")

    async with genai.Client(api_key=api_key).aio as aclient:
        total_responses = 0
        total_start = time.perf_counter()
        sem = asyncio.Semaphore(256)

        async def single_request(i):
            async with sem:
                return await aclient.models.generate_content(
                    model=NATIVE_MODEL,
                    contents="Hello, how are you?",
                    config={"thinking_config": {"thinking_budget": 0}},
                )

        for r in range(1, num_rounds + 1):
            print(f"\n── Round {r}/{num_rounds} ({num_requests} requests) [native SDK] ──")
            start = time.perf_counter()
            done_count = 0

            async def tracked_request(i):
                nonlocal done_count
                result = await single_request(i)
                done_count += 1
                progress(done_count, num_requests)
                return result

            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*[tracked_request(i) for i in range(num_requests)]),
                    timeout=60,
                )
            except asyncio.TimeoutError:
                print(f"\n  ERROR: Round {r} timed out after 60s — requests still hanging!")
                sys.exit(1)

            elapsed = time.perf_counter() - start
            total_responses += len(responses)
            print(f"\n  {len(responses)} responses in {elapsed:.1f}s ({len(responses)/elapsed:.1f} req/s)")

        total_elapsed = time.perf_counter() - total_start
        print(f"\n{'='*50}")
        print(f"Total: {total_responses} responses in {total_elapsed:.1f}s ({total_responses/total_elapsed:.1f} req/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-requests", type=int, default=500)
    parser.add_argument("-r", "--num-rounds", type=int, default=10)
    parser.add_argument("--native", action="store_true", help="Use Google GenAI SDK directly instead of llm_master_repo")
    args = parser.parse_args()
    if args.native:
        asyncio.run(main_native(args.num_requests, args.num_rounds))
    else:
        asyncio.run(main(args.num_requests, args.num_rounds))
