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

print(f"Image URL: {response.content}")
```

## Troubleshooting

If you encounter issues when integrating a new provider:

1. **API Compatibility**: Ensure the provider's API is truly OpenAI-compatible
2. **Error Handling**: Add appropriate error handling for provider-specific errors
3. **Rate Limiting**: Configure appropriate rate limits to avoid being blocked
4. **Logging**: Enable DEBUG-level logging to see detailed API interactions

## Best Practices

1. **Documentation**: Document the provider's capabilities and limitations
2. **Testing**: Create comprehensive tests for the new provider
3. **Fallbacks**: Implement fallback mechanisms for critical operations
4. **Monitoring**: Set up monitoring for API usage and errors 