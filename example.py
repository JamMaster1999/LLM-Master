from llm_master import QueryLLM, LLMConfig

# Initialize
config = LLMConfig.from_env()
llm = QueryLLM(config)

# Prepare messages
messages = [
    {
        "role": "user",
        "content": "Hello!",
        "image_paths": ["path/to/image.jpg"]  # Optional
    }
]

# Generate response with fallback and moderation
try:
    response = llm.query(
        model_name="claude-3-5-sonnet-latest",
        messages=messages,
        stream=True,
        fallback_provider="openai",
        fallback_model="gpt-4o",
        moderation=True
    )
    
    # Handle streaming response
    for chunk in response:
        print(chunk, end="", flush=True)
        
except Exception as e:
    print(f"Error: {str(e)}")