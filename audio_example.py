import asyncio
import base64
import os
from llm_master import QueryLLM, LLMConfig

async def run_audio_example():
    config = LLMConfig.from_env()
    llm = QueryLLM(config)
    
    messages = [
        {
            "role": "user", 
            "content": "Is a golden retriever a good family dog?",
        }
    ]

    try:
        # Query with audio output
        response = await llm.query(
            model_name="gpt-4o-audio-preview",
            messages=messages,
            stream=False,
            modality=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"}
        )
        
        # Print the text response
        print("Text response:")
        print(response.content)
        
        # Save the audio to a file if available
        if response.audio_data:
            wav_bytes = base64.b64decode(response.audio_data)
            output_file = "dog_response.wav"
            with open(output_file, "wb") as f:
                f.write(wav_bytes)
            print(f"\nAudio saved to '{output_file}'")
        else:
            print("\nNo audio data received in the response")
        
        # Print response metadata
        print("\n--- Response Metadata ---")
        print(f"Model: {response.model_name}")
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print(f"Cost: ${response.cost:.6f}")
        print(f"Latency: {response.latency:.2f} seconds")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_audio_example()) 