import time
import base64
import logging
import requests
import os
from pathlib import Path
from typing import Dict, Optional, List, Union, Any, Generator, AsyncGenerator
import asyncio

from .base_provider import UnifiedProvider, ProviderError
from .anthropic_provider import AnthropicProvider
from .bfl_provider import BFLProvider, ContentModerationError
from .classes import LLMResponse, LLMError, ModelRegistry, RateLimiter
from .config import LLMConfig

logger = logging.getLogger(__name__)

class ImageFormatError(LLMError):
    """Raised when there's an issue with image formatting"""
    pass

class QueryLLM:
    """Main class for handling LLM interactions with retry logic and fallbacks"""

    SUPPORTED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp']
    ERROR_REPORTING_URL = 'https://xh9i-rdvs-rnon.n7c.xano.io/api:9toeBRNq/error_logs'
    MAX_RETRIES = 3

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the QueryLLM handler

        Args:
            config: LLMConfig instance. If None, will attempt to load from environment
        """
        self.config = config or LLMConfig.from_env()
        self._providers = {}
        self._rate_limiters = {}
        
        # Initialize rate limiters for each model
        for model_name, model_config in ModelRegistry.CONFIGS.items():
            self._rate_limiters[model_name] = RateLimiter(model_config, model_name)
            
        logger.info("Initialized QueryLLM handler with rate limiters")

    async def _process_request(self, model_name: str, request: dict) -> LLMResponse:
        """Process a single request"""
        provider = self._get_provider(model_name)
        
        # Format messages for the provider
        formatted_messages = [
            self._format_message(msg, provider) for msg in request["messages"]
        ]
        
        # Get kwargs and remove moderation flag if present
        kwargs = request.get("kwargs", {}).copy()
        kwargs.pop("moderation", None)  # Remove moderation flag if present
        
        # Call the provider using async generate
        response = await provider.generate(
            messages=formatted_messages,
            model=model_name,
            **kwargs
        )
        
        # Track usage if available
        if hasattr(provider, 'last_usage') and provider.last_usage is not None:
            await self._rate_limiters[model_name].add_token_usage(
                provider.last_usage.input_tokens + provider.last_usage.output_tokens
            )
            
        return response

    async def query(self,
              model_name: str,
              messages: List[Dict[str, Any]],
              stream: bool = False,
              fallback_provider: Optional[str] = None,
              fallback_model: Optional[str] = None,
              moderation: bool = False,
              _is_fallback: bool = False,  # Internal parameter to prevent infinite recursion
              **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Queue and process a request while respecting rate limits
        
        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries
            stream: Whether to stream the response
            fallback_provider: Provider to fallback to if the primary fails
            fallback_model: Model to fallback to if the primary fails
            moderation: Whether to run content moderation
            _is_fallback: Internal parameter to prevent infinite recursion on fallbacks
            **kwargs: Additional model-specific parameters (e.g., temperature, reasoning_effort)
            
        Returns:
            Either a LLMResponse object or an AsyncGenerator for streaming
        """
        if stream:
            # Return the async generator directly without awaiting it
            return self._streaming_query(
                model_name=model_name,
                messages=messages,
                fallback_provider=fallback_provider if not _is_fallback else None,
                fallback_model=fallback_model if not _is_fallback else None,
                moderation=moderation,
                _is_fallback=_is_fallback,
                **kwargs
            )
            
        start_time = time.time()
        retry_count = 0
        rate_limiter = self._rate_limiters[model_name]

        # Content moderation if enabled
        if moderation:
            self._moderate_content(messages)

        while retry_count <= self.MAX_RETRIES:
            try:
                if retry_count > 0:
                    delay = 2 ** (retry_count - 1)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

                # Wait for rate limit capacity
                await rate_limiter.wait_for_capacity()
                
                # Process the request
                response = await self._process_request(model_name, {
                    "messages": messages,
                    "kwargs": {
                        "stream": stream,
                        "moderation": moderation,
                        **kwargs  # Pass through all model-specific parameters
                    }
                })
                
                # Calculate and set latency
                response.latency = time.time() - start_time
                return response

            except ProviderError as e:
                # Special handling for content policy violations
                if e.status_code == "CONTENT_POLICY" and fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"Content policy violation, falling back to {fallback_model}")
                    return await self.query(
                        model_name=fallback_model,
                        messages=messages,
                        stream=stream,
                        moderation=moderation,
                        _is_fallback=True,  # Mark as a fallback to prevent further fallbacks
                        **kwargs
                    )
                # Special handling for BFL content moderation errors (no retry)
                elif isinstance(e, ContentModerationError) or e.status_code == "BFL_CONTENT_MODERATED":
                    logger.warning("BFL content moderation error detected, not retrying")
                    if fallback_provider and fallback_model and not _is_fallback:
                        logger.info(f"Using fallback provider {fallback_provider} with model {fallback_model}")
                        return await self.query(
                            model_name=fallback_model,
                            messages=messages,
                            stream=stream,
                            moderation=moderation,
                            _is_fallback=True,
                            # Do NOT pass any kwargs to fallback provider for BFL fallbacks
                        )
                    else:
                        raise
                # Regular retry logic
                elif retry_count < self.MAX_RETRIES:
                    retry_count += 1
                    continue
                # Use fallback after retries are exhausted
                elif fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"All retries failed, falling back to {fallback_model}")
                    return await self.query(
                        model_name=fallback_model,
                        messages=messages,
                        stream=stream,
                        moderation=moderation,
                        _is_fallback=True,  # Mark as a fallback to prevent further fallbacks
                        **kwargs
                    )
                else:
                    raise

            except Exception as e:
                # Regular retry logic
                if retry_count < self.MAX_RETRIES:
                    retry_count += 1
                    continue
                # Use fallback after retries are exhausted
                elif fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"All retries failed, falling back to {fallback_model}")
                    return await self.query(
                        model_name=fallback_model,
                        messages=messages,
                        stream=stream,
                        moderation=moderation,
                        _is_fallback=True,  # Mark as a fallback to prevent further fallbacks
                        **kwargs
                    )
                else:
                    raise

    async def _streaming_query(self,
                         model_name: str,
                         messages: List[Dict[str, Any]],
                         fallback_provider: Optional[str] = None,
                         fallback_model: Optional[str] = None,
                         moderation: bool = False,
                         _is_fallback: bool = False,
                         **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Internal method to handle streaming. Returns an async generator of chunks.
        """
        start_time = time.time()
        retry_count = 0
        provider = self._get_provider(model_name)
        rate_limiter = self._rate_limiters[model_name]

        if moderation:
            self._moderate_content(messages)

        while retry_count <= self.MAX_RETRIES:
            try:
                if retry_count > 0:
                    delay = 2 ** (retry_count - 1)
                    await asyncio.sleep(delay)
                    logger.info(f"Retrying in {delay} seconds...")

                # Wait for rate limit capacity
                await rate_limiter.wait_for_capacity()

                # Format messages for the provider
                formatted_messages = [
                    self._format_message(msg, provider) for msg in messages
                ]

                # Use the provider in streaming mode
                full_response = ""
                # First await the coroutine to get the generator
                stream_generator = await provider.generate(
                    messages=formatted_messages,
                    model=model_name,
                    stream=True,
                    **kwargs
                )
                # Now iterate through the regular generator
                for chunk in stream_generator:
                    full_response += chunk
                    yield chunk  # <--- The actual streaming output to the caller

                # Post-stream, handle usage if available
                if hasattr(provider, 'last_usage') and provider.last_usage is not None:
                    await self._rate_limiters[model_name].add_token_usage(
                        provider.last_usage.input_tokens + provider.last_usage.output_tokens
                    )
                return

            except ProviderError as e:
                # Special handling for content policy violations  
                if e.status_code == "CONTENT_POLICY" and fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"Content policy violation, falling back to {fallback_model}")
                    
                    # Just use the main query method for fallback
                    fallback_generator = await self.query(
                        model_name=fallback_model,
                        messages=messages,
                        stream=True,
                        moderation=moderation,
                        _is_fallback=True,  # Mark as a fallback to prevent further fallbacks
                        **kwargs
                    )
                    
                    # Relay all chunks from the fallback
                    async for chunk in fallback_generator:
                        yield chunk
                    return
                # Regular retry logic
                elif retry_count < self.MAX_RETRIES:
                    retry_count += 1
                    continue
                # Use fallback after retries are exhausted  
                elif fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"All retries failed, falling back to {fallback_model}")
                    
                    # Just use the main query method for fallback
                    fallback_generator = await self.query(
                        model_name=fallback_model,
                        messages=messages,
                        stream=True,
                        moderation=moderation,
                        _is_fallback=True,  # Mark as a fallback to prevent further fallbacks
                        **kwargs
                    )
                    
                    # Relay all chunks from the fallback
                    async for chunk in fallback_generator:
                        yield chunk
                    return
                else:
                    raise
                    
            except Exception as e:
                # Regular retry logic
                if retry_count < self.MAX_RETRIES:
                    retry_count += 1
                    continue
                # Use fallback after retries are exhausted
                elif fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"All retries failed, falling back to {fallback_model}")
                    
                    # Just use the main query method for fallback
                    fallback_generator = await self.query(
                        model_name=fallback_model,
                        messages=messages,
                        stream=True,
                        moderation=moderation,
                        _is_fallback=True,  # Mark as a fallback to prevent further fallbacks
                        **kwargs
                    )
                    
                    # Relay all chunks from the fallback
                    async for chunk in fallback_generator:
                        yield chunk
                    return
                else:
                    raise

    def _moderate_content(self, messages: List[Dict[str, Any]]):
        """Perform content moderation on the last user message"""
        last_user_message = next(
            (msg for msg in reversed(messages) if msg["role"] == "user"), 
            None
        )
        if last_user_message:
            try:
                moderation_provider = UnifiedProvider("openai", self.config)
                formatted_message = self._format_message(last_user_message, moderation_provider)

                moderation_response = moderation_provider.moderate(
                    input=formatted_message["content"],
                    model="omni-moderation-latest"
                )

                if moderation_response["results"][0]["flagged"]:
                    flagged_categories = [
                        category for category, is_flagged in
                        moderation_response["results"][0]["categories"].items()
                        if is_flagged
                    ]
                    error_msg = f"Content moderation failed. Flagged categories: {', '.join(flagged_categories)}"
                    logger.error(error_msg)
                    raise LLMError(error_msg)

            except Exception as e:
                logger.error(f"Moderation failed: {str(e)}")
                raise LLMError(f"Moderation failed: {str(e)}")

    def _report_error(self, error: Exception, function: str = 'query_llm'):
        """Report errors to external logging service"""
        try:
            line_num = error.__traceback__.tb_lineno
            error_msg = f"Error: Line {line_num}; Description: {str(error)}"
        except:
            error_msg = str(error)
        try:
            requests.post(
                self.ERROR_REPORTING_URL,
                data={
                    'error': error_msg,
                    'function': function
                }
            )
            logger.info(f'Error: {error_msg}; has been recorded in database')
        except Exception as e:
            logger.error(f"Failed to report error: {str(e)}")

    def _format_message(self, message: Dict[str, Any], provider) -> Dict[str, Any]:
        """Helper function to format entire message content with text and/or images"""

        def encode_image(image_input: str) -> str:
            # If input is already base64
            if isinstance(image_input, str) and image_input.startswith('data:image/'):
                return image_input.split(',')[1]
            if isinstance(image_input, str) and ';base64,' in image_input:
                return image_input

            # If input is a URL
            if isinstance(image_input, str) and any(image_input.startswith(prefix) 
                for prefix in ['http://', 'https://']):
                try:
                    response = requests.get(image_input, timeout=10)
                    response.raise_for_status()
                    return base64.b64encode(response.content).decode('utf-8')
                except Exception as e:
                    raise ImageFormatError(f"Failed to fetch image from URL {image_input}: {str(e)}")
            
            # Otherwise treat as file path
            path_obj = Path(image_input)
            if not path_obj.exists():
                raise ImageFormatError(f"Image file not found: {path_obj}")
            with open(path_obj, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        def get_mime_type(image_input: str) -> str:
            """Helper to determine mime type from various input formats"""
            if isinstance(image_input, str):
                # From data URI
                if image_input.startswith('data:image/'):
                    mime_type = image_input.split(';')[0].split('/')[1]
                # From URL
                elif any(image_input.startswith(prefix) for prefix in ['http://', 'https://']):
                    try:
                        response = requests.head(image_input, timeout=5)
                        content_type = response.headers.get('content-type', '')
                        if content_type.startswith('image/'):
                            mime_type = content_type.split('/')[1]
                        else:
                            # Fallback to extension if content-type is not helpful
                            mime_type = Path(image_input).suffix.lower()[1:]
                    except:
                        # Fallback to extension if request fails
                        mime_type = Path(image_input).suffix.lower()[1:]
                # From file path
                else:
                    mime_type = Path(image_input).suffix.lower()[1:]
            else:
                mime_type = 'png'  # Default fallback
            
            # Normalize jpg to jpeg
            if mime_type == 'jpg':
                mime_type = 'jpeg'
            
            return f"image/{mime_type}"

        # If already has 'content', assume it's well-structured
        if 'content' in message:
            return message

        text = message.get('text')
        # Support image_paths, direct base64 images, and URLs
        images = message.get('image_paths', message.get('images', []))
        image_details = message.get('image_details', ['auto'] * len(images))

        if not images:
            return {"role": message["role"], "content": text}

        content = []
        if text:
            content.append({"type": "text", "text": text})

        for img, detail in zip(images, image_details):
            try:
                base64_image = encode_image(img)
                mime_type = get_mime_type(img)
                
                if mime_type not in self.SUPPORTED_MIME_TYPES:
                    raise ImageFormatError(
                        f"Unsupported image type: {mime_type}. "
                        f"Must be one of: {', '.join(self.SUPPORTED_MIME_TYPES)}"
                    )

                # Format differently for Anthropic vs. OpenAI/Gemini
                if isinstance(provider, AnthropicProvider):
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_image
                        }
                    })
                else:
                    # OpenAI/Gemini format
                    image_data = {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                    # Only add detail for OpenAI provider
                    if isinstance(provider, UnifiedProvider) and provider.provider == "openai":
                        image_data["detail"] = detail
                    
                    content.append({
                        "type": "image_url",
                        "image_url": image_data
                    })
            except Exception as e:
                logger.error(f"Error formatting image {img}: {str(e)}")
                raise ImageFormatError(f"Failed to format image {img}: {str(e)}")

        return {"role": message["role"], "content": content}

    def _get_provider(self, model_name: str) -> Union[UnifiedProvider, AnthropicProvider, BFLProvider]:
        """
        Get or create appropriate provider for the model

        Args:
            model_name: Name of the model to use

        Returns:
            Provider instance

        Raises:
            LLMError: If the provider is not supported
        """
        provider_map = {
            "gpt": ("openai", UnifiedProvider),
            "o3": ("openai", UnifiedProvider),  # Added mapping for o3 models
            "claude": ("anthropic", AnthropicProvider),
            "gemini": ("gemini", UnifiedProvider),
            "mistral": ("mistral", UnifiedProvider),
            "recraft": ("recraft", UnifiedProvider),
            "fireworks": ("fireworks", UnifiedProvider),
            "imagen": ("gemini", UnifiedProvider),  # Use gemini provider for imagen models
            "flux": (None, BFLProvider)  # Use "flux" instead of "bfl" to match BFL model names
        }

        try:
            provider_key = next(
                (key for key in provider_map if key in model_name.lower()),
                None
            )
            if not provider_key:
                raise LLMError(f"Unsupported model: {model_name}")

            if provider_key not in self._providers:
                provider_name, provider_class = provider_map[provider_key]
                if provider_class == UnifiedProvider:
                    self._providers[provider_key] = provider_class(provider_name, self.config)
                else:
                    self._providers[provider_key] = provider_class(self.config)

            return self._providers[provider_key]

        except Exception as e:
            logger.error(f"Error getting provider: {str(e)}")
            raise LLMError(f"Failed to get provider: {str(e)}")

    def stop_generation(self):
        """Stop the current generation if any"""
        if self._providers:
            for provider in self._providers.values():
                provider.stop_generation()