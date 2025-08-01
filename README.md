# LLM Master

## Fallback Configuration

Use `fallback_config` to specify which parameters should be passed to fallback models:

```python
# Primary model with advanced params, fallback gets only compatible ones
response = await llm.query(
    model_name="o3",
    reasoning_effort="high",  # o3-specific parameter
    temperature=0.7,
    fallback_model="claude-3-5-sonnet-latest",
    fallback_config={"temperature": 0.7}  # Only pass temperature to Claude
)

# Safe fallback: no parameters passed to fallback model
response = await llm.query(
    model_name="gemini-2.5-flash", 
    reasoning_effort="low",  # Gemini-specific
    fallback_model="gpt-4.1"
    # No fallback_config = no params passed to fallback (safe default)
)
```

---

# Adding a New Provider Using OpenAI Base Client

This guide walks through the process of adding a new API provider (like Recraft.AI) that uses the OpenAI client base class but with a different endpoint and potentially different features.

## Overview

Many AI service providers are now implementing OpenAI-compatible APIs, allowing developers to use the OpenAI client libraries with different backends. This guide explains how to integrate such a provider into the LLM Master system.

## Prerequisites

- Access to the API provider's endpoint and API key
- Understanding of the provider's API capabilities and limitations

## Step-by-Step Integration Guide

### 1. Update the PROVIDER_CONFIGS Dictionary

Add your new provider to the `PROVIDER_CONFIGS` dictionary in the `UnifiedProvider` class in `base_provider.py`:

```python
PROVIDER_CONFIGS = {
    # ... existing providers
    "new_provider": {
        "client_class": OpenAI,
        "base_url": "https://api.your-provider.com/v1",
        "api_key_attr": "your_provider_api_key",
        "supports_caching": False,  # Set to True if the provider supports caching
        "generate_map": {
            "model-name": "_generate_custom_function"  # Map models to custom generators
        }
    }
}
```

### 2. Implement Custom Generation Methods

For models that require special handling (like image generation), add custom generator methods to the `UnifiedProvider` class:

```python
async def _generate_custom_function(self, messages, model, **kwargs):
    """Custom generator for specific model"""
    # Extract parameters from messages or kwargs
    param = kwargs.get("param") or extract_from_messages(messages)
    
    # Call the appropriate API endpoint
    response = await self._call_api_endpoint(param, model, **kwargs)
    
    # Return formatted response
    return LLMResponse(
        content=response.data,
        model_name=model,
        usage=Usage(...),
        latency=0.0
    )
```

### 3. Update the LLMConfig Class

Add your provider's API key to the `LLMConfig` class in `config.py`:

```python
@dataclass
class LLMConfig:
    # ... existing API keys
    new_provider_api_key: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        return cls(
            # ... existing keys
            new_provider_api_key=os.getenv("NEW_PROVIDER_API_KEY")
        )
    
    def validate(self) -> Dict[str, bool]:
        return {
            # ... existing providers
            "new_provider": bool(self.new_provider_api_key)
        }
```

### 4. Add Models to the ModelRegistry

Register your provider's models in the `ModelRegistry` class in `classes.py`:

```python
CONFIGS = {
    # ... existing models
    "new-provider-model-name": ModelConfig(
        input_price_per_million=0.00,  # Set appropriate pricing
        output_price_per_million=0.00,  # Set appropriate pricing
        cached_input_price_per_million=None,  # If applicable
        rate_limit_rpm=100  # Set appropriate rate limit
    )
}
```

### 4.1 Model Naming Conventions

When adding models to the registry, consider the provider mapping in `_get_provider()`. This method uses substring matching to identify which provider to use:

```python
provider_key = next(
    (key for key in provider_map if key in model_name.lower()),
    None
)
```

This means your model names should contain a substring that matches a key in the provider map. For example:
- Models with "gpt" in the name will use the OpenAI provider
- Models with "claude" in the name will use the Anthropic provider

If your provider has model names that don't clearly indicate the provider (e.g., models that don't contain the provider name), you have two options:

1. Use a substring that appears in all model names:
   ```python
   provider_map = {
       # ... existing mappings
       "common-substring": (None, CustomProvider)  # Will match any model containing "common-substring"
   }
   ```

2. Add a prefix to your model names when registering them:
   ```python
   CONFIGS = {
       # ... existing models
       "prefix-model-name": ModelConfig(...)  # Adding a provider-specific prefix
   }
   ```
   
   And then use that prefix in your API calls:
   ```python
   response = await llm.query(
       model_name="prefix-model-name",  # Use with prefix
       messages=[{"role": "user", "content": "Your prompt here"}]
   )
   ```

Choose the approach that best fits your use case and model naming conventions.

### 5. Update the Provider Mapping

Update the provider mapping in `QueryLLM._get_provider` method in `response_synthesizer.py`:

```python
provider_map = {
    # ... existing mappings
    "new_provider_prefix": ("new_provider", UnifiedProvider)
}
```

This tells the system which prefix maps to which provider type and provider class.

### 6. For Completely Different APIs (Non-OpenAI Compatible)

For APIs that don't follow the OpenAI structure, create a completely separate provider class:

```python
class CustomProvider(BaseLLMProvider):
    """Provider for a custom API"""
    
    def __init__(self, config=None):
        self.config = config or LLMConfig()
        # Initialize any custom clients or sessions
        
    async def generate(self, messages, model, **kwargs):
        # Implement custom generation logic
        # ...
```

Then add it to the provider mapping:

```python
provider_map = {
    # ... existing mappings
    "custom_prefix": (None, CustomProvider)  # First parameter not needed
}
```

#### 6.1 Important considerations for custom providers:

1. **Proper Error Handling**: Use the `ProviderError` class for API-specific errors and `ConfigurationError` for setup issues
   
2. **Asynchronous Implementation**: All provider methods should be properly asynchronous to avoid blocking the event loop:
   ```python
   # For blocking API calls, use ThreadPoolExecutor:
   loop = asyncio.get_event_loop()
   with ThreadPoolExecutor() as executor:
       result = await loop.run_in_executor(
           executor,
           lambda: requests.post(
               api_url,
               headers=headers,
               json=payload
           ).json()
       )
   ```

3. **Streaming Support**: Clearly document whether streaming is supported and raise appropriate errors if it's not:
   ```python
   async def generate(self, messages, model, stream=False, **kwargs):
       if stream:
           raise ProviderError("Streaming not supported for this provider")
       # ...
   ```

4. **Provider Mapping**: When adding the provider to `_get_provider()` in `response_synthesizer.py`, ensure the key matches a substring in your model names:
   ```python
   provider_map = {
       # ... existing mappings
       "model_prefix": (None, CustomProvider)  # Use a prefix that appears in all your model names
   }
   ```

5. **Standardized Response Format**: Always return responses in the standardized `LLMResponse` format:
   ```python
   return LLMResponse(
       content=result_content,  # The actual response content
       model_name=model,
       usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out),
       latency=latency
   )
   ```

### 7. Test Your Integration

Test your integration by setting the appropriate environment variables and making requests to your provider:

```python
import os
from llm_master import QueryLLM

# Set environment variable
os.environ["NEW_PROVIDER_API_KEY"] = "your-api-key"

# Create LLM instance
llm = QueryLLM()

# Make a request
response = await llm.query(
    model_name="new-provider-model-name",
    messages=[{"role": "user", "content": "Your prompt here"}]
)

print(response.content)
```

## Example: Adding Recraft.AI for Image Generation

Here's a concrete example of adding Recraft.AI for image generation:

### Provider Configuration

```python
"recraft": {
    "client_class": OpenAI,
    "base_url": "https://external.api.recraft.ai/v1",
    "api_key_attr": "recraft_api_key",
    "supports_caching": False,
    "generate_map": {
        "recraft-image-gen": "_generate_recraft_image"
    }
}
```

### Custom Generation Method

```python
async def _generate_recraft_image(self, messages, model, **kwargs):
    # Extract prompt from messages or kwargs
    prompt = kwargs.get("prompt") or extract_prompt(messages)
    style = kwargs.get("style", "digital_illustration")
    
    # Call the OpenAI client's images.generate method
    response = await loop.run_in_executor(
        None,
        lambda: self.client.images.generate(
            prompt=prompt,
            style=style,
            **kwargs
        )
    )
    
    # Return formatted response
    return LLMResponse(
        content=response.data[0].url,
        model_name=model,
        usage=Usage(input_tokens=0, output_tokens=0),
        latency=0.0
    )
```

### Usage Example

```python
response = await llm.query(
    model_name="recraft-image-gen",
    prompt="race car on a track",
    style="digital_illustration"
)

print(response.content)
```

## Example: Adding BFL for Image Generation with Polling

Here's an example of adding Black Forest Labs (BFL) image generation API that uses polling:

### Custom Provider Class

```python
class BFLProvider(BaseLLMProvider):
    """Provider for Black Forest Labs (BFL) image generation API"""
    
    def __init__(self, config=None):
        self.config = config or LLMConfig.from_env()
        
        # Get API key from config
        api_key = getattr(self.config, "bfl_api_key")
        if not api_key:
            raise ConfigurationError("Missing BFL API key")
            
        self.api_key = api_key
        self.base_url = "https://api.us1.bfl.ai/v1"
        self._current_generation = None
        self.last_usage = None
        self.last_response = None
```

### Generate Method with Polling

```python
async def generate(self, 
              messages: List[Dict[str, Any]], 
              model: str,
              stream: bool = False,
              **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
    """Generate an image using the BFL API"""
    if stream:
        raise ProviderError("Streaming is not supported for BFL image generation")
    
    # Extract prompt from messages or kwargs
    prompt = kwargs.get("prompt")
    if not prompt and messages:
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                prompt = msg["content"]
                break
    
    # Get image dimensions
    width = kwargs.get("width", 1024)
    height = kwargs.get("height", 768)
    
    start_time = time.time()
    
    try:
        # Initial request to generate image
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            request_data = await loop.run_in_executor(
                executor,
                lambda: requests.post(
                    f"{self.base_url}/{model}",
                    headers={
                        'accept': 'application/json',
                        'x-key': self.api_key,
                        'Content-Type': 'application/json',
                    },
                    json={
                        'prompt': prompt,
                        'width': width,
                        'height': height,
                    },
                ).json()
            )
        
        request_id = request_data["id"]
        
        # Poll for result
        while True:
            with ThreadPoolExecutor() as executor:
                poll_result = await loop.run_in_executor(
                    executor,
                    lambda: requests.get(
                        f"{self.base_url}/get_result",
                        headers={
                            'accept': 'application/json',
                            'x-key': self.api_key,
                        },
                        params={
                            'id': request_id,
                        },
                    ).json()
                )
            
            if poll_result.get("status") == "Ready":
                image_url = poll_result["result"]["sample"]
                latency = time.time() - start_time
                usage = Usage(input_tokens=0, output_tokens=0)
                return LLMResponse(
                    content=image_url,
                    model_name=model,
                    usage=usage,
                    latency=latency
                )
            
            # Wait before polling again
            await asyncio.sleep(0.5)
            
    except Exception as e:
        raise ProviderError(f"BFL image generation failed: {str(e)}")
```

### Provider Mapping

```python
provider_map = {
    # ... existing mappings
    "flux": (None, BFLProvider)  # Will match "flux-pro-1.1"
}
```

### Usage Example

```python
response = await llm.query(
    model_name="flux-pro-1.1",
    messages=[{"role": "user", "content": "A cat running with a fish in a market"}],
    width=1024,  # Optional
    height=768   # Optional
)

print(response.content)
```

## Troubleshooting

If you encounter issues when integrating a new provider:

1. **API Compatibility**: Ensure the provider's API is truly OpenAI-compatible if you're using the UnifiedProvider

2. **Error Handling**: Add appropriate error handling for provider-specific errors

3. **Rate Limiting**: Configure appropriate rate limits to avoid being blocked

4. **Logging**: Enable DEBUG-level logging to see detailed API interactions:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

5. **"Unsupported model" Error**: This common error occurs when the provider mapping can't match your model name. Check:
   - Is your model name registered in `ModelRegistry.CONFIGS`?
   - Does your model name contain a substring that matches a key in the provider map?
   - If using a custom provider, is your provider properly registered in `_get_provider()` method?

   Example error:
   ```
   ERROR: Error getting provider: Unsupported model: your-model-name
   ```
   
   Solution: Either change your model name to include the provider key, or update the provider mapping to match your model name pattern.

## Best Practices

1. **Documentation**: Document the provider's capabilities and limitations
2. **Testing**: Create comprehensive tests for the new provider
3. **Fallbacks**: Implement fallback mechanisms for critical operations
4. **Monitoring**: Set up monitoring for API usage and errors 