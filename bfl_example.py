import os
import asyncio
from dotenv import load_dotenv
from llm_master import QueryLLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

async def generate_image():
    """Example of generating an image with BFL"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Ensure the BFL API key is set
    bfl_api_key = os.environ.get("BFL_API_KEY")
    if not bfl_api_key:
        print("Error: BFL_API_KEY environment variable not set.")
        print("Please add it to your .env file or set it in your environment.")
        return
    else:
        print(f"Using BFL API key: {bfl_api_key[:4]}...{bfl_api_key[-4:]}")
    
    # Create the LLM client
    llm = QueryLLM()
    
    # Check available providers
    print(f"Available providers: {llm.config.validate()}")
    
    # Define the prompt
    prompt = "A cat on its back legs running like a human is holding a big silver fish with its arms. The cat is running away from the shop owner and has a panicked look on his face. The scene is situated in a crowded market."
    
    try:
        print(f"Attempting to generate image using model 'flux-pro-1.1'...")
        # Generate the image
        response = await llm.query(
            model_name="flux-pro-1.1",
            messages=[{"role": "user", "content": prompt}],
            width=1024,  # Optional: specify image width
            height=768   # Optional: specify image height
        )
        
        # Print the result (which is the image URL)
        print(f"Image generation successful! URL: {response.content}")
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(generate_image()) 