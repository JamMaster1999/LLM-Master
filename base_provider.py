from typing import List, Dict, Any, Union, Generator, Optional
from openai import OpenAI as StandardOpenAI
from posthog import Posthog
from posthog.ai.openai import OpenAI
from .classes import BaseLLMProvider, LLMResponse, Usage, RateLimiter, ModelConfig, ModelRegistry
from .config import LLMConfig
import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import re

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
    """Provider for OpenAI-compatible APIs"""
    
    PROVIDER_CONFIGS = {
        "openai": {
            "client_class": OpenAI,
            "base_url": None,
            "api_key_attr": "openai_api_key",
            "supports_caching": True,
            "generate_map": {
                "gpt-image-1": "_generate_recraft_image"
            }
        },
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
        "perplexity": {
            "client_class": OpenAI,
            "base_url": "https://api.perplexity.ai",
            "api_key_attr": "perplexity_api_key",
            "supports_caching": False,
            "generate_map": {}
        }
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
        # Extract modalities and audio parameters if provided
        modalities = kwargs.pop('modality', None)
        audio_params = kwargs.pop('audio', None)
        
        # Extract PostHog parameters if provided as a dict
        posthog_params = kwargs.pop('posthog', None)
        if posthog_params and isinstance(posthog_params, dict):
            # Unpack PostHog parameters with posthog_ prefix
            for key, value in posthog_params.items():
                kwargs[f'posthog_{key}'] = value
        
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
        
        # Unified usage handling for all OpenAI-compatible providers
        cached_tokens = 0
        if self.supports_caching:
            try:
                # Access cached_tokens from prompt_tokens_details as per OpenAI docs
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
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
            
        # Extract citations if they exist in the response (specifically for Perplexity API)
        citations = None
        if self.provider == "perplexity":
            # Access the citations directly from the response object
            try:
                if hasattr(response, 'citations'):
                    citations = response.citations
                    self.last_citations = citations
                    
                    # Format citations in the requested format
                    citation_text = "<citation_list>"
                    for citation in citations:
                        citation_text += f"<citation_source>{citation}</citation_source>"
                    citation_text += "</citation_list>"
                    
                    # Append the citations to the content
                    content += f"\n{citation_text}"
            except Exception as e:
                logger.error(f"Error extracting citations: {str(e)}")
                # Continue despite the error
        
        return LLMResponse(
            content=content,
            model_name=model,
            usage=usage,
            latency=0,  # Will be set by ResponseSynthesizer
            audio_data=audio_data,
            citations=citations
        )

    def _stream_response(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        try:
            model = kwargs.pop('model', None)
            full_response = ""  # Initialize before the loop
            
            # Extract PostHog parameters if provided as a dict
            posthog_params = kwargs.pop('posthog', None)
            if posthog_params and isinstance(posthog_params, dict):
                # Unpack PostHog parameters with posthog_ prefix
                for key, value in posthog_params.items():
                    kwargs[f'posthog_{key}'] = value
            
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )
            
            self._current_generation = stream
            
            self.last_citations = None # Ensure reset for the stream
            self.last_images = None    # Ensure reset for the stream

            def replace_citation_link(match):
                try:
                    number = int(match.group(1))
                    index = number - 1  # Citations are 1-indexed in the text
                    if self.last_citations and 0 <= index < len(self.last_citations):
                        citation = self.last_citations[index]
                        url_link = getattr(citation, 'url', citation if isinstance(citation, str) else "#")
                        obfuscated_url = url_link.replace("https://", "h_ttps://").replace("http://", "h_ttp://")
                        return f"<inline_citation>{number}:{obfuscated_url}</inline_citation>"
                except (ValueError, IndexError, AttributeError, TypeError) as e:
                    logger.warning(f"Failed to replace citation link for {match.group(0)}: {e}")
                return match.group(0)
            
            for chunk in stream:
                # Check for usage information in the chunk (final chunk or usage-only chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    # This is a chunk with usage information
                    
                    cached_tokens = 0
                    if self.supports_caching:
                        try:
                            # Access cached_tokens from prompt_tokens_details as per OpenAI docs
                            if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details:
                                cached_tokens = getattr(chunk.usage.prompt_tokens_details, 'cached_tokens', 0)
                        except AttributeError:
                            pass
                    
                    self.last_usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        cached_tokens=cached_tokens
                    )

                # --- Perplexity Specific Handling --- 
                if self.provider == "perplexity":
                    # 1. Store Citations when found
                    if hasattr(chunk, 'citations') and chunk.citations:
                        if self.last_citations is None: # Store only the first time they appear
                             logger.info("Storing citations found in stream chunk.")
                             self.last_citations = list(chunk.citations)
                        
                    # 2. Store Images when found 
                    if hasattr(chunk, 'images') and chunk.images:
                        if self.last_images is None: # Store only the first time they appear
                            logger.info("Storing images found in stream chunk.") 
                            self.last_images = list(chunk.images)

                    # 3. Process and yield Content
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        processed_content = re.sub(r'\[(\d+)\]', replace_citation_link, content)
                        full_response += processed_content
                        yield processed_content

                # --- Handling for other OpenAI compatible providers (non-perplexity content) ---
                elif chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Store the full response for potential later use
            self.last_response = full_response
            
            # Yield final Citation List for Perplexity
            if self.provider == "perplexity" and self.last_citations:
                logger.info("Yielding final citation list.")
                citation_text = "<citation_list>\n"
                for citation in self.last_citations:
                    citation_source = getattr(citation, 'url', citation if isinstance(citation, str) else str(citation))
                    # Obfuscate citation URLs   
                    citation_text += f"<citation_source><citation_url>{citation_source}</citation_url></citation_source>\n"
                citation_text += "</citation_list>"
                yield f"\n{citation_text}\n" # Add surrounding newlines

            # Yield final Image List for Perplexity
            if self.provider == "perplexity" and self.last_images:
                logger.info("Yielding final image list.")
                image_list_text = "<image_citation_list>\n"
                for img_data in self.last_images:
                    if isinstance(img_data, dict):
                        img_url = img_data.get('image_url', '#')
                        origin_url = img_data.get('origin_url', '#')
                        # Obfuscate image URLs  
                        image_list_text += "<image_citation>\n"
                        image_list_text += f"<image_url>{img_url}</image_url>\n"
                        image_list_text += f"<origin_url>{origin_url}</origin_url>\n"
                        image_list_text += "</image_citation>\n"
                    else:
                        logger.warning(f"Unexpected image data format in final list: {img_data}")
                image_list_text += "</image_citation_list>"
                yield f"\n{image_list_text}\n" # Add surrounding newlines

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
            self._current_generation.close()
            self._current_generation = None
    
    def _setup_client(self, provider_config: Dict):
        """Set up the provider client with configuration"""
        # Get API key from config
        api_key = getattr(self.config, provider_config["api_key_attr"])
        if not api_key:
            raise ConfigurationError(f"Missing API key for provider: {self.provider}")
            
        try:
            self.posthog = Posthog(
                project_api_key=os.getenv("POSTHOG_API_KEY", "phc_1uBDKATKfxK7ougGiL9F9hnCgeXJvc4k6TMP2oekfnK"),
                host=os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
            )
            
            self.client = OpenAI(
                base_url=provider_config["base_url"],
                api_key=api_key,
                posthog_client=self.posthog
            )
            logger.info(f"Initialized {self.provider} provider with base URL: {provider_config['base_url']}")
                
            self.supports_caching = provider_config["supports_caching"]
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            self.last_citations = None
            
            logger.info(f"Successfully initialized {self.provider} provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize {self.provider} provider: {str(e)}")

    async def _generate_recraft_image(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> LLMResponse:
        """Generate an image using either Recraft or GPT Image generation APIs
        
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
        is_gpt_image = "gpt-image" in model.lower()
        
        # Set model-specific parameters
        if is_gpt_image:
            # GPT Image model returns b64_json by default
            style_params = {}
            logger.info(f"Generating image with GPT Image provider. Base URL: {self.client.base_url}")
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
            
            # Extract usage information if available (for GPT Image model)
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage_obj = response.usage
                usage = Usage(
                    input_tokens=getattr(usage_obj, 'prompt_tokens', 0),
                    output_tokens=getattr(usage_obj, 'completion_tokens', 0),
                    cached_tokens=0
                )
            else:
                # Default usage for models that don't provide it
                usage = Usage(input_tokens=0, output_tokens=1)
            
            # Return response with the appropriate content format
            content = None
            if is_gpt_image:
                content = response.data[0].b64_json
            else:
                content = response.data[0].url
            
            return LLMResponse(
                content=content,
                model_name=model,
                usage=usage,
                latency=latency
            )
            
        except Exception as e:
            provider_name = "GPT Image" if is_gpt_image else "Recraft"
            logger.error(f"Error generating image with {provider_name}: {str(e)}")
            raise ProviderError(f"Failed to generate image: {str(e)}")