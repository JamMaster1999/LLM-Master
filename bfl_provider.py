import os
import requests
import asyncio
import time
import logging
from typing import List, Dict, Any, Union, Generator, Optional
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

from .classes import BaseLLMProvider, LLMResponse, Usage
from .config import LLMConfig
from .base_provider import ProviderError, ConfigurationError

# Set up logging
logger = logging.getLogger(__name__)

# Special exception for content moderation that should not be retried
class ContentModerationError(ProviderError):
    """Exception raised when content is moderated and should not be retried"""
    def __init__(self, message: str):
        super().__init__(message, status_code="BFL_CONTENT_MODERATED")

class BFLProvider(BaseLLMProvider):
    """Provider for Black Forest Labs (BFL) image generation API"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the BFL provider
        
        Args:
            config: LLMConfig instance. If None, will attempt to load from environment
        """
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
        
        logger.info("Successfully initialized BFL provider")

    async def generate(self, 
                messages: List[Dict[str, Any]], 
                model: str,
                stream: bool = False,
                **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        """
        Generate an image using the BFL API
        
        Args:
            messages: List of message dictionaries
            model: Model name
            stream: Whether to stream the response (not supported for BFL)
            **kwargs: Additional arguments to pass to the API
                - width: Image width (default: 1024)
                - height: Image height (default: 768)
                - timeout: Maximum time to wait for generation in seconds (default: 120)
        
        Returns:
            LLMResponse containing the image URL
        """
        if stream:
            raise ProviderError("Streaming is not supported for BFL image generation")
        
        # Extract prompt from messages or kwargs
        prompt = kwargs.get("prompt")
        if not prompt and messages:
            # Get the last user message content
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    prompt = msg["content"]
                    break
        
        if not prompt:
            raise ProviderError("No prompt provided for image generation")
        
        # Get image dimensions
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 768)
        timeout = kwargs.get("timeout", 120)  # Default timeout of 2 minutes
        
        start_time = time.time()
        
        try:
            # Make initial request to generate image
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
                            **{k: v for k, v in kwargs.items() if k not in ['prompt', 'width', 'height', 'timeout', 'fallback_provider', 'fallback_model']}
                        },
                        timeout=30,  # Request timeout
                    ).json()
                )
            
            if "id" not in request_data:
                raise ProviderError(f"Failed to initiate BFL image generation: {request_data}")
            
            request_id = request_data["id"]
            logger.info(f"BFL image generation initiated with request ID: {request_id}")
            
            # Poll for result with timeout
            deadline = start_time + timeout
            poll_count = 0
            
            async def poll_once():
                with ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(
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
                            timeout=10,  # Request timeout
                        ).json()
                    )
            
            while time.time() < deadline:
                poll_count += 1
                poll_result = await poll_once()
                
                if poll_result.get("status") == "Ready":
                    logger.info(f"BFL image generation completed after {poll_count} polls")
                    result = poll_result
                    break
                elif poll_result.get("status") == "Failed":
                    raise ProviderError(f"BFL image generation failed: {poll_result}")
                elif poll_result.get("status") == "Content Moderated":
                    logger.warning(f"BFL image generation rejected due to content moderation")
                    
                    # Check if fallback provider and model are specified
                    fallback_provider = kwargs.get("fallback_provider")
                    fallback_model = kwargs.get("fallback_model")
                    
                    if fallback_provider and fallback_model:
                        logger.info(f"Attempting fallback to {fallback_provider} with model {fallback_model}")
                        
                        try:
                            # Dynamically import the provider module
                            module_name = f".{fallback_provider}_provider"
                            provider_module = import_module(module_name, package=__package__)
                            
                            # Get the provider class (convention: ProviderNameProvider)
                            provider_class_name = f"{fallback_provider.title()}Provider"
                            provider_class = getattr(provider_module, provider_class_name)
                            
                            # Initialize provider with same config
                            fallback_provider_instance = provider_class(self.config)
                            
                            # Generate using fallback - only pass essential parameters
                            # Don't pass BFL-specific parameters that might not be supported by the fallback
                            return await fallback_provider_instance.generate(
                                messages=messages,
                                model=fallback_model,
                                stream=False  # Always use non-streaming for fallback
                            )
                        except Exception as e:
                            logger.error(f"Fallback to {fallback_provider} failed: {str(e)}")
                            raise ContentModerationError(f"BFL content moderated and fallback failed: {str(e)}")
                    else:
                        raise ContentModerationError(
                            "BFL image generation rejected due to content moderation. Specify fallback_provider and fallback_model to use a different provider."
                        )
                    
                    # No need to continue polling, content moderation won't change
                    break
                
                logger.debug(f"BFL image poll {poll_count}: Status={poll_result.get('status')}")
                
                # Wait before polling again (500ms)
                await asyncio.sleep(0.5)
            else:
                # Timeout reached
                raise ProviderError(f"BFL image generation timed out after {timeout} seconds")
            
            # Extract image URL
            if "result" in result and "sample" in result["result"]:
                image_url = result["result"]["sample"]
                
                # Calculate latency
                latency = time.time() - start_time
                
                # Create usage object (BFL doesn't provide token counts)
                usage = Usage(input_tokens=0, output_tokens=1)
                
                # Store response for later reference
                self.last_response = result
                self.last_usage = usage
                
                logger.info(f"BFL image generated successfully in {latency:.2f} seconds")
                
                return LLMResponse(
                    content=image_url,
                    model_name=model,
                    usage=usage,
                    latency=latency
                )
            else:
                raise ProviderError(f"Invalid response from BFL: {result}")
            
        except requests.RequestException as e:
            logger.error(f"Network error during BFL image generation: {str(e)}")
            raise ProviderError(f"Network error with BFL API: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout during BFL image generation")
            raise ProviderError(f"BFL image generation timed out")
        except ContentModerationError:
            # Re-raise content moderation errors without wrapping them
            raise
        except Exception as e:
            logger.error(f"Error during BFL image generation: {str(e)}")
            raise ProviderError(f"BFL image generation failed: {str(e)}")

    def stop_generation(self):
        """Stop the current generation (not supported for BFL)"""
        # BFL doesn't support stopping generation once started
        pass 