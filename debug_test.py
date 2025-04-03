import time, base64, json, requests, asyncio
import os
import sys
from typing import List, Dict, Union, Any 
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

# Load environment variables from .env file
load_dotenv()

# Import our providers
from llm_master import QueryLLM, LLMConfig

async def debug_test():
    # Print loaded env vars for debugging
    print(f"API Keys loaded: GEMINI_API_KEY={'*****' if os.getenv('GEMINI_API_KEY') else 'NOT FOUND'}")
    print(f"API Keys loaded: OPENAI_API_KEY={'*****' if os.getenv('OPENAI_API_KEY') else 'NOT FOUND'}")
    
    config = LLMConfig.from_env()
    llm = QueryLLM(config)
    
    messages = [
        {
            "role": "user", 
            "content": "What is 2+2?",  # Simple query to reduce processing time
        },
    ]

    try:
        print("Starting non-streaming test...")
        
        # Try the non-streaming mode
        response = await llm.query(
            model_name="gemini-2.0-flash",
            messages=messages,
            stream=False,
            fallback_provider="openai",
            fallback_model="gpt-4o",
            moderation=False
        )
        
        # Print response data
        print("\nResponse content:", response.content)
        print("\n--- Response Metadata ---")
        print(f"Model: {response.model_name}")
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print(f"Latency: {response.latency:.2f} seconds")
        
        # Now try streaming mode for comparison
        print("\n\nStarting streaming test...")
        stream_gen = await llm.query(
            model_name="gemini-2.0-flash",
            messages=messages,
            stream=True,
            fallback_provider="openai",
            fallback_model="gpt-4o",
            moderation=False
        )
        
        full_response = ""
        print("Streaming response: ", end="", flush=True)
        async for chunk in stream_gen:
            full_response += chunk
            print(chunk, end="", flush=True)
            
        print("\nStreaming complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# Run in asyncio loop
if __name__ == "__main__":
    asyncio.run(debug_test()) 