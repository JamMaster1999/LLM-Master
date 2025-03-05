import os 
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
# Initialize
import time, base64, json, requests
import asyncio
from typing import List, Dict, Union, Any 

# Import our providers
from llm_master import QueryLLM, LLMConfig

async def main():
    config = LLMConfig.from_env()
    llm = QueryLLM(config)

    messages = [
        {
            "role": "user", 
            "content": "Hi What is your name? ",
        },
    ]

    try:
        # The query method returns a coroutine that resolves to an async generator
        response_generator = await llm.query(
            model_name="gemini-2.0-flash",
            messages=messages,
            stream=True,
            fallback_provider="openai",
            fallback_model="gpt-4o",
            moderation=False
        )
        
        # Iterate through the async generator
        async for chunk in response_generator:
            print(chunk, end="", flush=True)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())