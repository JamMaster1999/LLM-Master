#!/usr/bin/env python3
"""
LLM Master Example Script
Demonstrates non-streaming, streaming, and usage tracking across different providers.

To run this script:
1. cd /path/to/uflo-AI-server
2. source ai-server/bin/activate  
3. python -m llm_master.example

Make sure you have API keys set in your .env file:
- GEMINI_API_KEY
- OPENAI_API_KEY  
- PERPLEXITY_API_KEY (optional)
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath('..'))
load_dotenv()

from llm_master import QueryLLM, LLMConfig

async def test_model(llm, model_name, test_name):
    """Test a model with both streaming and non-streaming"""
    print(f"\n{'='*60}")
    print(f"Testing {test_name} ({model_name})")
    print(f"{'='*60}")
    
    messages = [{"role": "user", "content": "Tell me a very short joke."}]
    
    # Test non-streaming
    print(f"\n1. Non-streaming {test_name}:")
    try:
        response = await llm.query(
            model_name=model_name,
            messages=messages,
            stream=False
        )
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Cost: ${response.cost:.6f}")
        print(f"Latency: {response.latency:.2f}s")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test streaming
    print(f"\n2. Streaming {test_name}:")
    try:
        stream = await llm.query(
            model_name=model_name,
            messages=messages,
            stream=True
        )
        
        full_content = ""
        print("Response: ", end="", flush=True)
        async for chunk in stream:
            full_content += chunk
            print(chunk, end="", flush=True)
        
        # Check for usage info after streaming
        provider = llm._get_provider(model_name)
        if hasattr(provider, 'last_usage') and provider.last_usage:
            print(f"\nUsage: {provider.last_usage}")
            from llm_master.classes import ModelRegistry
            try:
                model_config = ModelRegistry.get_config(model_name)
                cost = provider.last_usage.calculate_cost(model_config)
                print(f"Cost: ${cost:.6f}")
            except Exception as e:
                print(f"Cost calculation failed: {e}")
        else:
            print("\n⚠️  No usage information available after streaming")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def test_audio(llm):
    """Test audio generation with OpenAI"""
    print(f"\n{'='*60}")
    print("Testing Audio Generation (OpenAI)")
    print(f"{'='*60}")
    
    messages = [{"role": "user", "content": "Is a golden retriever a good family dog?"}]
    
    try:
        response = await llm.query(
            model_name="gpt-4o-audio-preview",
            messages=messages,
            stream=False,
            modality=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"}
        )
        
        print(f"Text response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Cost: ${response.cost:.6f}")
        print(f"Latency: {response.latency:.2f}s")
        
        # Save audio if available
        if response.audio_data:
            import base64
            wav_bytes = base64.b64decode(response.audio_data)
            output_file = "dog_response.wav"
            with open(output_file, "wb") as f:
                f.write(wav_bytes)
            print(f"🎵 Audio saved to '{output_file}'")
        else:
            print("⚠️  No audio data received")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Main test function"""
    print("🚀 LLM Master Test Suite")
    print("Checking API keys...")
    
    # Check API keys
    api_keys = {
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "PERPLEXITY_API_KEY": bool(os.getenv("PERPLEXITY_API_KEY"))
    }
    
    for key, present in api_keys.items():
        status = "✅" if present else "❌"
        print(f"{status} {key}: {'Present' if present else 'Missing'}")
    
    # Initialize LLM
    config = LLMConfig.from_env()
    llm = QueryLLM(config)
    
    # Test different models
    tests = [
        ("gemini-2.5-flash", "Gemini"),
        ("gpt-4o-mini", "OpenAI"),
    ]
    
    # Add Perplexity if API key is available
    if api_keys["PERPLEXITY_API_KEY"]:
        tests.append(("sonar", "Perplexity"))
    
    for model_name, test_name in tests:
        await test_model(llm, model_name, test_name)
    
    # Test audio generation if OpenAI key is available
    # if api_keys["OPENAI_API_KEY"]:
    #     await test_audio(llm)
    
    print(f"\n{'='*60}")
    print("🎉 All tests completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())