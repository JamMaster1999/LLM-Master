#!/usr/bin/env python
"""
Simple test script for Perplexity API citations handling
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import llm_master
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Print environment variables for debugging
print(f"PERPLEXITY_API_KEY present: {bool(os.environ.get('PERPLEXITY_API_KEY'))}")

from llm_master import QueryLLM, LLMConfig

async def test_perplexity():
    # Initialize with config from environment
    config = LLMConfig.from_env()
    llm = QueryLLM(config)
    
    # Sample messages
    messages = [
        {
            "role": "system",
            "content": "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, polite conversation with a user."
        },
        {   
            "role": "user",
            "content": "How many stars are in the universe?"
        },
    ]
    
    # Test non-streaming query
    print("Testing non-streaming Perplexity API with citations")
    response = await llm.query(
        model_name="sonar",
        messages=messages,
        stream=False
    )
    
    print(f"Response content: {response.content}")
    print(f"Citations: {response.citations}")
    
    # Test streaming query
    print("\nTesting streaming Perplexity API with citations")
    stream_generator = await llm.query(
        model_name="sonar",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    citations_found = False
    
    async for chunk in stream_generator:
        if chunk.startswith("\n<citations>"):
            print(f"Found citations in stream: {chunk}")
            citations_found = True
        else:
            full_response += chunk
            print(f"Received chunk: {chunk}")
    
    print(f"\nFull response: {full_response[:100]}...")
    
    # After streaming is complete, check if provider has citations
    provider = llm._get_provider("sonar")
    if hasattr(provider, 'last_citations') and provider.last_citations:
        print(f"Citations from provider.last_citations: {provider.last_citations}")
        citations_found = True
    
    if not citations_found:
        print("No citations found in streaming response")




if __name__ == "__main__":
    asyncio.run(test_perplexity())
