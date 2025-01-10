import os
from llm_master import QueryLLM, LLMConfig

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
os.environ["MISTRAL_API_KEY"] = "your-mistral-api-key"

import time, base64, json
from typing import List, Dict, Union, Any

# Import our providers
from base_provider import UnifiedProvider
from anthropic_provider import AnthropicProvider
from llm_master.classes import LLMResponse

# Initialize providers
gemini_provider = UnifiedProvider("gemini")
openai_provider = UnifiedProvider("openai")
mistral_provider = UnifiedProvider("mistral")
anthropic_provider = AnthropicProvider()

def format_message_content(provider, text: str = None, image_paths: List[str] = None) -> Union[str, List[Dict]]:
    """Helper function to format message content with text and/or images"""
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    if not image_paths:
        return text
        
    content = []
    if text:
        content.append({"type": "text", "text": text})
    
    for path in image_paths:
        base64_image = encode_image(path)
        mime_type = f"image/{path.split('.')[-1].lower()}"
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        elif mime_type not in ['image/jpeg', 'image/png', 'image/gif', 'image/webp']:
            raise ValueError(f"Unsupported image type: {mime_type}. Must be one of: image/jpeg, image/png, image/gif, image/webp")
        
        if isinstance(provider, AnthropicProvider):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_image
                }
            })
        else:  # OpenAI/Gemini format
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })
    
    return content


def print_response_details(response: LLMResponse):
    def to_dict(obj):
        # Handle None and basic types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Try to convert object to dictionary
        try:
            # Get object attributes
            obj_dict = vars(obj)
            # Recursively convert nested objects
            return {key: to_dict(value) for key, value in obj_dict.items()}
        except:
            # If conversion fails, return string representation
            return str(obj)
    
    # Convert to dictionary and then to JSON
    response_dict = to_dict(response)
    print(json.dumps(response_dict, indent=2))
    return response_dict


def query_llm(provider, model_name: str, messages: List[Dict[str, Any]], stream: bool = False):
    start_time = time.time()

    try:
        if stream:
            full_response = ""
            for chunk in provider.generate(messages=messages, model=model_name, stream=True):
                full_response += chunk
                print(chunk, end="", flush=True)
            content = full_response
        else:
            response = provider.generate(messages=messages, model=model_name)
            content = response.content
        
        # Create LLMResponse object for both streaming and non-streaming
        if hasattr(provider, 'last_usage'):
            response = LLMResponse(
                content=content,
                model_name=model_name,
                usage=provider.last_usage,
                latency=time.time() - start_time
            )
            # print("\n\nResponse details:" if stream else "")
            # print_response_details(response)
            return response  # Return the response object
        else:
            print("\nNo usage details available")
            return content  # Return the content if no usage details
            
    except Exception as e:
        print(f"\nError during {'streaming' if stream else 'generation'}: {e}")
        
    finally:
        print(f"\nTotal time: {time.time() - start_time:.2f}s")


# provider = anthropic_provider
# model_name = "claude-3-5-sonnet-latest"
# messages = [
#     {
#         "role": "user", 
#         "content": format_message_content(
#             provider=provider,
#             text="What's in this image?",
#             image_paths=["/Users/sina/Downloads/Uflo Platform/extract_pdf/llm_supported_docs/ezgif-3-a03c1be7a6.jpg"]
#         )
#     },
#     {
#         "role": "assistant",
#         "content": "The image shows a man with glasses smiling."
#     },
#     {
#         "role": "user",
#         "content": format_message_content(
#             provider=provider,
#             text="Can you tell me the background of the previous image? What does this new image show?",
#             image_paths=["/Users/sina/Downloads/Uflo Platform/extract_pdf/llm_supported_docs/Acid-Base Solutions screenshot.png"]
#         )
#     }
# ]
# response = query_llm(provider, model_name, messages, stream=True)
# response_json = print_response_details(response)