from typing import List, Dict, Any, Union, Generator, Optional
from openai import OpenAI
# from mistralai import Mistral
import google.generativeai as genai
from .classes import BaseLLMProvider, LLMResponse, Usage, RateLimiter, ModelConfig, ModelRegistry
from .config import LLMConfig
import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProviderError(Exception):
    """Base exception for provider-related errors"""
    def __init__(self, message: str, status_code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code

class ConfigurationError(ProviderError):
    """Raised when there's an issue with provider configuration"""
    pass

class UnifiedProvider(BaseLLMProvider):
    """Provider for OpenAI, Gemini, and Mistral models using synchronous clients"""
    
    PROVIDER_CONFIGS = {
        "openai": {
            "client_class": OpenAI,
            "base_url": None,
            "api_key_attr": "openai_api_key",
            "supports_caching": True,
            "generate_map": {}  # Default generation functions
        },
        "gemini": {
            "client_class": OpenAI,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key_attr": "gemini_api_key",
            "supports_caching": False,
            "generate_map": {
                "imagen-3.0-generate-002": "_generate_recraft_image"
            }
        },
        # "mistral": {
        #     "client_class": Mistral,
        #     "base_url": None,
        #     "api_key_attr": "mistral_api_key",
        #     "supports_caching": False,
        #     "generate_map": {}
        # },
        "recraft": {
            "client_class": OpenAI,
            "base_url": "https://external.api.recraft.ai/v1",
            "api_key_attr": "recraft_api_key",
            "supports_caching": False,
            "generate_map": {
                "recraftv3": "_generate_recraft_image"
            }
        },
        "fireworks": {
            "client_class": OpenAI,
            "base_url": "https://api.fireworks.ai/inference/v1",
            "api_key_attr": "fireworks_api_key",
            "supports_caching": False,
            "generate_map": {}
        },

    }

    def __init__(self, provider: str = "openai", config: Optional[LLMConfig] = None):
        """Initialize the provider with configuration"""
        super().__init__()
        self.provider = provider
        self.config = config or LLMConfig()
        
        # Initialize rate limiters dictionary but don't create them yet
        self.rate_limiters = {}
        
        self.openai_compatible_providers = [
            p for p, cfg in self.PROVIDER_CONFIGS.items() 
            if cfg["client_class"] == OpenAI
        ]
        
        # Initialize provider client
        provider_config = self.PROVIDER_CONFIGS.get(provider)
        if not provider_config:
            raise ConfigurationError(f"Unsupported provider: {provider}")
            
        self._setup_client(provider_config)
        
    def _get_rate_limiter(self, model_name: str) -> RateLimiter:
        """
        Get or create a rate limiter for the model
        
        Args:
            model_name: Name of the model to get/create rate limiter for
            
        Returns:
            RateLimiter instance
        """
        if model_name not in self.rate_limiters:
            # Create rate limiter on demand
            model_config = ModelRegistry.get_config(model_name)
            self.rate_limiters[model_name] = RateLimiter(
                model_config=model_config,
                model_name=model_name
            )
        
        return self.rate_limiters[model_name]
        
    async def generate_parallel(self, requests: List[Dict], model: str, **kwargs) -> List[LLMResponse]:
        """Generate responses in parallel while respecting rate limits"""
        rate_limiter = self._get_rate_limiter(model)
        responses = []
        
        async def process_request(request):
            await rate_limiter.wait_for_capacity()
            return await self.generate_async(request, **kwargs)
            
        # Process requests in parallel
        tasks = [process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        return responses
        
    async def generate_async(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Async version of generate"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: self.generate(messages, **kwargs)
            )
        return response

    def _count_gemini_tokens(self, messages: List[Dict[str, Any]], model: str, gemini_model=None) -> int:
        """Count tokens for Gemini models
        
        Args:
            messages: List of message dictionaries
            model: Model name
            gemini_model: Optional pre-initialized model
            
        Returns:
            Total token count
        """
        if gemini_model is None:
            gemini_model = genai.GenerativeModel(f"models/{model}")
        
        total_tokens = 0
        for msg in messages:
            if isinstance(msg['content'], str):
                # Text only
                count = gemini_model.count_tokens(msg['content'])
                total_tokens += count.total_tokens
            elif isinstance(msg['content'], list):
                # Multimodal content
                text_parts = []
                image_count = 0
                
                for part in msg['content']:
                    if part.get('type') == 'text':
                        text_parts.append(part['text'])
                    elif part.get('type') == 'image_url':  # Handle OpenAI/Gemini format
                        image_count += 1
                    elif part.get('type') == 'image':  # Handle Anthropic format
                        image_count += 1
                
                # Add fixed token count for each image
                total_tokens += image_count * 258
                
                # Count tokens for text parts if any
                if text_parts:
                    count = gemini_model.count_tokens(" ".join(text_parts))
                    total_tokens += count.total_tokens
        
        return total_tokens

    async def generate(self, 
            messages: List[Dict[str, Any]], 
            model: str,
            stream: bool = False,
            **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        """Async generate method"""
        # Check if there's a custom generator for this model
        provider_config = self.PROVIDER_CONFIGS.get(self.provider, {})
        custom_generator = provider_config.get("generate_map", {}).get(model)
        
        if custom_generator:
            # Call the custom generator
            generator_method = getattr(self, custom_generator)
            return await generator_method(messages, model, **kwargs)
        
        # Standard generation flow
        if stream:
            return self._stream_response(messages, model=model, **kwargs)
        
        loop = asyncio.get_event_loop()
        if self.provider in self.openai_compatible_providers:
            # Extract modalities and audio parameters if provided
            modalities = kwargs.pop('modality', None)
            audio_params = kwargs.pop('audio', None)
            
            # Create completion parameters
            completion_params = {
                'model': model,
                'messages': messages,
                'stream': False,
                **kwargs
            }
            
            # Add modalities and audio parameters if they exist
            if modalities:
                completion_params['modalities'] = modalities
            if audio_params:
                completion_params['audio'] = audio_params
            
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(**completion_params)
            )
            
            # Check for Gemini content policy violation
            if self.provider == "gemini":
                if not response or not response.choices or not response.choices[0].message:
                    raise ProviderError(
                        "Gemini content policy violation detected",
                        status_code="CONTENT_POLICY"
                    )
                
                # Initialize Gemini model for token counting
                gemini_model = genai.GenerativeModel(f"models/{model}")
                
                # Count input tokens including images
                input_tokens = await loop.run_in_executor(
                    None,
                    lambda: self._count_gemini_tokens(messages, model, gemini_model)
                )
                
                # Count output tokens
                output_count = await loop.run_in_executor(
                    None,
                    lambda: gemini_model.count_tokens(response.choices[0].message.content)
                )
                
                usage = Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_count.total_tokens,
                    cached_tokens=0
                )
            else:
                # OpenAI usage handling
                cached_tokens = 0
                if self.supports_caching:
                    try:
                        cached_tokens = getattr(response.usage, 'cached_tokens', 0)
                    except AttributeError:
                        pass
                
                usage = Usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cached_tokens=cached_tokens
                )
            
            content = response.choices[0].message.content
            
            # Extract audio data if it exists in the response
            audio_data = None
            if hasattr(response.choices[0].message, 'audio') and response.choices[0].message.audio:
                audio_data = response.choices[0].message.audio.data
        
        else:  # mistral
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.complete(
                    model=model,
                    messages=messages,
                    **kwargs
                )
            )
            
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached_tokens=0
            )
            
            content = response.choices[0].message.content
            audio_data = None
        
        return LLMResponse(
            content=content,
            model_name=model,
            usage=usage,
            latency=0,  # Will be set by ResponseSynthesizer
            audio_data=audio_data
        )

    def _stream_response(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        try:
            if self.provider in self.openai_compatible_providers:
                model = kwargs.pop('model', None)
                full_response = ""  # Initialize before the loop
                
                if self.provider == "gemini":
                    # Initialize Gemini model for token counting
                    gemini_model = genai.GenerativeModel(f"models/{model}")
                    
                    # Count input tokens including images
                    input_tokens = self._count_gemini_tokens(messages, model, gemini_model)
                
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    #! error for gemini might be here stream_options={"include_usage": True} if self.provider in ["openai", "fireworks"] else None,
                    stream_options={"include_usage": True} if self.provider in self.openai_compatible_providers else None,
                    **kwargs
                )
                
                self._current_generation = stream
                
                for chunk in stream:
                    if self.provider in self.openai_compatible_providers:
                    #! error for gemini might be here if self.provider in ["openai", "fireworks"]:
                        # OpenAI usage handling
                        if not chunk.choices and hasattr(chunk, 'usage') and chunk.usage:
                            self.last_usage = Usage(
                                input_tokens=chunk.usage.prompt_tokens,
                                output_tokens=chunk.usage.completion_tokens,
                                cached_tokens=0
                            )
                        elif chunk.choices and chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield content
                    else:  # gemini
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield content
                
                # For Gemini, calculate output tokens after streaming is complete
                if self.provider == "gemini":
                    output_count = gemini_model.count_tokens(full_response)
                    self.last_usage = Usage(
                        input_tokens=input_tokens,
                        output_tokens=output_count.total_tokens,
                        cached_tokens=0
                    )
                
                # Store the full response for potential later use
                self.last_response = full_response
                    
            else:  # mistral
                model = kwargs.pop('model', None)
                stream = self.client.chat.stream(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                
                self._current_generation = stream
                full_response = ""  # Initialize before the loop
                
                for chunk in stream:
                    # Check if this is the final chunk with usage data
                    if hasattr(chunk.data, 'usage') and chunk.data.usage:
                        self.last_usage = Usage(
                            input_tokens=chunk.data.usage.prompt_tokens,
                            output_tokens=chunk.data.usage.completion_tokens,
                            cached_tokens=0
                        )
                    # Regular content chunk
                    elif chunk.data.choices[0].delta.content:
                        content = chunk.data.choices[0].delta.content
                        full_response += content
                        yield content
                
                # Store the full response for potential later use
                self.last_response = full_response
                    
        finally:
            self._current_generation = None

    def moderate(self, input: Union[str, List[Dict[str, Any]]], model: str) -> Dict[str, Any]:
        """Moderate content using OpenAI's moderation endpoint"""
        if self.provider != "openai":
            # Create a temporary OpenAI client for moderation
            config = self.PROVIDER_CONFIGS["openai"]
            api_key = os.getenv(config["api_key_env"])
            client = config["client_class"](api_key=api_key)
        else:
            client = self.client
        
        # If input is a string, convert to the expected format
        if isinstance(input, str):
            input = [{"type": "text", "text": input}]
        
        response = client.moderations.create(
            model=model,
            input=input
        )
        
        return response.model_dump()

    def stop_generation(self):
        """Stop the current generation if any"""
        if self._current_generation:
            if self.provider in self.openai_compatible_providers:
                self._current_generation.close()
            self._current_generation = None

    def _setup_client(self, provider_config: Dict):
        """Set up the provider client with configuration"""
        # Get API key from config
        api_key = getattr(self.config, provider_config["api_key_attr"])
        if not api_key:
            raise ConfigurationError(f"Missing API key for provider: {self.provider}")
            
        try:
            if self.provider == "gemini":
                genai.configure(api_key=api_key)
                
            if provider_config["client_class"] == OpenAI:
                # Use the OpenAI client for all providers that use it
                self.client = provider_config["client_class"](
                    base_url=provider_config["base_url"],
                    api_key=api_key
                )
                logger.info(f"Initialized {self.provider} provider with base URL: {provider_config['base_url']}")
            else:  # For clients that don't use base_url
                self.client = provider_config["client_class"](
                    api_key=api_key
                )
                
            self.supports_caching = provider_config["supports_caching"]
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            
            logger.info(f"Successfully initialized {self.provider} provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize {self.provider} provider: {str(e)}")

    async def _generate_recraft_image(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> LLMResponse:
        """Generate an image using either Recraft or Imagen image generation APIs
        
        Args:
            messages: List of message dictionaries (will extract prompt from the last user message)
            model: The model to use for generation
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            LLMResponse containing the image URL or base64 data
        """
        # Extract prompt from messages or from kwargs
        prompt = kwargs.get("prompt")
        
        # If prompt not in kwargs, try to extract from messages
        if not prompt and messages:
            for message in reversed(messages):  # Start from the most recent message
                if message.get("role") == "user" and message.get("content"):
                    prompt = message.get("content")
                    break
        
        if not prompt:
            raise ProviderError("No prompt provided for image generation")
        
        # Determine which model we're using
        is_imagen = "imagen" in model.lower()
        
        # Set model-specific parameters
        if is_imagen:
            # For Imagen models - MUST use b64_json format
            style_params = {"response_format": "b64_json"}
            logger.info(f"Generating image with Imagen provider. Base URL: {self.client.base_url}")
        else:
            # For Recraft models
            style_params = {"style": kwargs.get("style", "digital_illustration")}
            logger.info(f"Generating image with Recraft provider. Base URL: {self.client.base_url}")
            
        try:
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            # Call the OpenAI client's images.generate method
            response = await loop.run_in_executor(
                None,
                lambda: self.client.images.generate(
                    prompt=prompt,
                    model=model,
                    n=kwargs.get("n", 1),
                    **{**style_params, **{k: v for k, v in kwargs.items() 
                       if k not in ["prompt", "style", "n"]}}
                )
            )
            
            latency = time.time() - start_time
            
            # Return response with the appropriate content format
            return LLMResponse(
                content=response.data[0].b64_json if is_imagen else response.data[0].url,
                model_name=model,
                usage=Usage(input_tokens=0, output_tokens=1),
                latency=latency
            )
            
        except Exception as e:
            provider_name = "Imagen" if is_imagen else "Recraft"
            logger.error(f"Error generating image with {provider_name}: {str(e)}")
            raise ProviderError(f"Failed to generate image: {str(e)}")