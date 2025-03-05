import os
import requests
import asyncio
import time
import logging
from typing import List, Dict, Any, Union, Generator, Optional
from concurrent.futures import ThreadPoolExecutor

from .classes import BaseLLMProvider, LLMResponse, Usage
from .config import LLMConfig
from .base_provider import ProviderError, ConfigurationError

# Set up logging
logger = logging.getLogger(__name__)

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
                            'safety_tolerance': 6,
                            **{k: v for k, v in kwargs.items() if k not in ['prompt', 'width', 'height']}
                        },
                    ).json()
                )
            
            if "id" not in request_data:
                raise ProviderError(f"Failed to initiate BFL image generation: {request_data}")
            
            request_id = request_data["id"]
            
            # Poll for result
            result = None
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
                    result = poll_result
                    break
                elif poll_result.get("status") == "Failed":
                    raise ProviderError(f"BFL image generation failed: {poll_result}")
                
                # Wait before polling again
                await asyncio.sleep(0.5)
            
            # Extract image URL
            if result and "result" in result and "sample" in result["result"]:
                image_url = result["result"]["sample"]
                
                # Calculate latency
                latency = time.time() - start_time
                
                # Create usage object (BFL doesn't provide token counts)
                usage = Usage(input_tokens=0, output_tokens=1)
                
                # Store response for later reference
                self.last_response = result
                self.last_usage = usage
                
                return LLMResponse(
                    content=image_url,
                    model_name=model,
                    usage=usage,
                    latency=latency
                )
            else:
                raise ProviderError(f"Invalid response from BFL: {result}")
            
        except Exception as e:
            logger.error(f"Error during BFL image generation: {str(e)}")
            raise ProviderError(f"BFL image generation failed: {str(e)}")

    def stop_generation(self):
        """Stop the current generation (not supported for BFL)"""
        # BFL doesn't support stopping generation once started
        pass 