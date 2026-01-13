import time
import base64
import logging
import requests
import os
from pathlib import Path
from typing import Dict, Optional, List, Union, Any, Generator, AsyncGenerator, Tuple
import asyncio

from .base_provider import UnifiedProvider, ProviderError
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .bfl_provider import BFLProvider, ContentModerationError
from .gemini_provider import GoogleGenAIProvider
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
    PROVIDER_SETUP = {
        "google_genai": (GoogleGenAIProvider, ["googleai:", "vertexai:"]),
        "openai_provider": (OpenAIProvider, ["responses-"]),
        "bfl_provider": (BFLProvider, ["flux-"]),
        "anthropic": (AnthropicProvider, ["claude-"]),
        "openai": (UnifiedProvider, ["gpt-", "o", "chatgpt-"]),
        "mistral": (UnifiedProvider, ["mistral-"]),
        "recraft": (UnifiedProvider, ["recraftv3"]),
        "fireworks": (UnifiedProvider, ["accounts/fireworks/models/"]),
        "perplexity": (UnifiedProvider, ["sonar"])
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._providers = {}
        self._rate_limiters = {}
        logger.info("Initialized QueryLLM handler")

    def _get_rate_limiter(self, model_name: str) -> RateLimiter:
        if model_name not in self._rate_limiters:
            model_config = ModelRegistry.get_config(model_name)
            self._rate_limiters[model_name] = RateLimiter(model_config, model_name)
        return self._rate_limiters[model_name]

    async def _try_fallback(self, fallback_model: str, messages: List[Dict[str, Any]], stream: bool, moderation: bool, fallback_config: Optional[Dict[str, Any]], **kwargs) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        return await self.query(
            model_name=fallback_model,
            messages=messages,
            stream=stream,
            moderation=moderation,
            _is_fallback=True,
            **(fallback_config or {}),
        )

    async def query(self, model_name: str, messages: List[Dict[str, Any]], stream: bool = False, fallback_provider: Optional[str] = None, fallback_model: Optional[str] = None, fallback_config: Optional[Dict[str, Any]] = None, moderation: bool = False, _is_fallback: bool = False, **kwargs) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        if stream:
            return self._streaming_query(model_name, messages, fallback_provider if not _is_fallback else None, fallback_model if not _is_fallback else None, fallback_config if not _is_fallback else None, moderation, _is_fallback, **kwargs)
            
        start_time = time.time()
        rate_limiter = self._get_rate_limiter(model_name)
        if moderation:
            self._moderate_content(messages)
        provider = self._get_provider(model_name)
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                await rate_limiter.wait_for_capacity()
                formatted_messages = [self._format_message(msg, provider) for msg in messages]
                response = await provider.generate(messages=formatted_messages, model=model_name, stream=False, **kwargs)
                response.latency = time.time() - start_time
                return response
            except ProviderError as e:
                if e.status_code == "CONTENT_POLICY" and fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"Content policy violation, falling back to {fallback_model}")
                    return await self._try_fallback(fallback_model, messages, stream, moderation, fallback_config, **kwargs)
                if isinstance(e, ContentModerationError) or e.status_code == "BFL_CONTENT_MODERATED":
                    logger.warning("BFL content moderation error detected, not retrying")
                    if fallback_provider and fallback_model and not _is_fallback:
                        logger.info(f"Using fallback provider {fallback_provider} with model {fallback_model}")
                        return await self._try_fallback(fallback_model, messages, stream, moderation, fallback_config, **kwargs)
                    raise
                if attempt >= self.MAX_RETRIES:
                    if fallback_provider and fallback_model and not _is_fallback:
                        logger.warning(f"All retries failed, falling back to {fallback_model}")
                        return await self._try_fallback(fallback_model, messages, stream, moderation, fallback_config, **kwargs)
                    raise

    async def async_query(
        self,
        model_name: str,
        requests: List[Dict[str, Any]],
        *,
        concurrency: Optional[int] = None,
    ) -> List[LLMResponse]:
        """High-throughput variant for processing multiple requests concurrently."""
        if not requests:
            return []

        provider = self._get_provider(model_name)
        rate_limiter = self._get_rate_limiter(model_name)

        # Fallback to individual query calls if provider does not support native async.
        if not getattr(provider, "supports_native_async", False):
            return await asyncio.gather(
                *[
                    self.query(
                        model_name=model_name,
                        messages=req.get("messages", []),
                        stream=False,
                        fallback_provider=req.get("fallback_provider"),
                        fallback_model=req.get("fallback_model"),
                        fallback_config=req.get("fallback_config"),
                        moderation=req.get("moderation", False),
                        **req.get("kwargs", {}),
                    )
                    for req in requests
                ]
            )

        max_concurrency = concurrency or 128
        rpm = rate_limiter.model_config.rate_limit_rpm or max_concurrency
        results: List[Optional[LLMResponse]] = [None] * len(requests)
        pending = list(enumerate(requests))
        async def process_request(index: int, request: Dict[str, Any]) -> Tuple[int, LLMResponse]:
            local_messages = request.get("messages", [])
            moderation = request.get("moderation", False)

            if moderation:
                self._moderate_content(local_messages)

            formatted_messages = request.get("formatted_messages")
            if formatted_messages is None:
                formatted_messages = [
                    self._format_message(message, provider) for message in local_messages
                ]

            kwargs = dict(request.get("kwargs", {}))
            fallback_provider = request.get("fallback_provider")
            fallback_model = request.get("fallback_model")
            fallback_config = request.get("fallback_config")
            
            async def try_fallback() -> Tuple[int, LLMResponse]:
                fb_response = await self.query(model_name=fallback_model, messages=local_messages, stream=False, fallback_provider=None, fallback_model=None, fallback_config=fallback_config, moderation=moderation)
                return index, fb_response

            last_error: Optional[Exception] = None
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    response = await provider.generate(messages=formatted_messages, model=model_name, stream=False, _native_async=True, **kwargs)
                    return index, response
                except ProviderError as exc:
                    last_error = exc
                    # Immediate fallback on content policy violation
                    if exc.status_code == "CONTENT_POLICY" and fallback_provider and fallback_model:
                        return await try_fallback()
                    # Fallback after exhausting retries
                    if attempt >= self.MAX_RETRIES and fallback_provider and fallback_model:
                        return await try_fallback()
                    if attempt >= self.MAX_RETRIES:
                        raise
                except Exception as exc:
                    last_error = exc
                    logger.exception("Unexpected error during async_query attempt %d for %s", attempt + 1, model_name)
                
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
            
            # Final fallback attempt or raise error
            if fallback_provider and fallback_model:
                return await try_fallback()
            raise last_error if isinstance(last_error, ProviderError) else ProviderError("All retries exhausted in async_query") from last_error

        while pending:
            batch_size = min(len(pending), rpm)
            batch = pending[:batch_size]
            pending = pending[batch_size:]

            await rate_limiter.wait_for_capacity(batch_size)
            semaphore = asyncio.Semaphore(min(max_concurrency, batch_size))

            async def gated_process(entry: Tuple[int, Dict[str, Any]]) -> Tuple[int, LLMResponse]:
                index, request = entry
                async with semaphore:
                    return await process_request(index, request)

            batch_results = await asyncio.gather(*(gated_process(entry) for entry in batch))

            for index, response in batch_results:
                results[index] = response

        return [response for response in results if response is not None]

    async def _iterate_stream(self, stream_generator) -> AsyncGenerator[str, None]:
        if hasattr(stream_generator, '__aiter__'):
            async for chunk in stream_generator:
                yield chunk
        else:
            for chunk in stream_generator:
                yield chunk

    async def _streaming_query(self, model_name: str, messages: List[Dict[str, Any]], fallback_provider: Optional[str] = None, fallback_model: Optional[str] = None, fallback_config: Optional[Dict[str, Any]] = None, moderation: bool = False, _is_fallback: bool = False, **kwargs) -> AsyncGenerator[str, None]:
        provider = self._get_provider(model_name)
        rate_limiter = self._get_rate_limiter(model_name)
        if moderation:
            self._moderate_content(messages)

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = 2 ** (attempt - 1)
                    await asyncio.sleep(delay)
                    logger.info(f"Retrying in {delay} seconds...")
                await rate_limiter.wait_for_capacity()
                formatted_messages = [self._format_message(msg, provider) for msg in messages]
                stream_generator = await provider.generate(messages=formatted_messages, model=model_name, stream=True, **kwargs)
                async for chunk in self._iterate_stream(stream_generator):
                    yield chunk
                return
            except ProviderError as e:
                if e.status_code == "CONTENT_POLICY" and fallback_provider and fallback_model and not _is_fallback:
                    logger.warning(f"Content policy violation, falling back to {fallback_model}")
                    async for chunk in await self._try_fallback(fallback_model, messages, True, moderation, fallback_config, **kwargs):
                        yield chunk
                    return
                if attempt >= self.MAX_RETRIES:
                    if fallback_provider and fallback_model and not _is_fallback:
                        logger.warning(f"All retries failed, falling back to {fallback_model}")
                        async for chunk in await self._try_fallback(fallback_model, messages, True, moderation, fallback_config, **kwargs):
                            yield chunk
                        return
                    raise
            except Exception:
                if attempt >= self.MAX_RETRIES:
                    if fallback_provider and fallback_model and not _is_fallback:
                        logger.warning(f"All retries failed, falling back to {fallback_model}")
                        async for chunk in await self._try_fallback(fallback_model, messages, True, moderation, fallback_config, **kwargs):
                            yield chunk
                        return
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

        def build_image_part(image_input: str, detail: str = "auto") -> Dict[str, Any]:
            if isinstance(provider, OpenAIProvider):
                # For Responses API, use data URIs directly or construct them
                if isinstance(image_input, str) and (
                    image_input.startswith("data:image/") or 
                    image_input.startswith("http://") or 
                    image_input.startswith("https://")
                ):
                    data_uri = image_input
                else:
                    base64_image = encode_image(image_input)
                    mime_type = get_mime_type(image_input)
                    data_uri = f"data:{mime_type};base64,{base64_image}"
                return {"type": "input_image", "image_url": data_uri}

            base64_image = encode_image(image_input)
            mime_type = get_mime_type(image_input)

            if isinstance(provider, AnthropicProvider):
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image
                    }
                }

            image_payload = {"url": f"data:{mime_type};base64,{base64_image}"}
            if isinstance(provider, UnifiedProvider) and provider.provider == "openai":
                image_payload["detail"] = detail

            return {"type": "image_url", "image_url": image_payload}

        def build_text_part(text_value: str, role: str = "user") -> Dict[str, Any]:
            if isinstance(provider, OpenAIProvider):
                # Responses API: user messages use input_text, assistant messages use output_text
                text_type = "output_text" if role == "assistant" else "input_text"
            else:
                text_type = "text"
            return {"type": text_type, "text": text_value}

        # Check if 'content' is already a list of parts
        current_message_content = message.get('content')
        if isinstance(current_message_content, list):
            # Transform parts to provider-specific format
            formatted_parts = []
            for part in current_message_content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")

                # Handle text parts
                if part_type in ("text", "input_text", "output_text"):
                    formatted_parts.append(build_text_part(part.get("text", ""), message["role"]))
                # Handle image parts
                elif part_type in ("image_url", "input_image"):
                    image_data = part.get("image_url")
                    if isinstance(image_data, dict):
                        image_data = image_data.get("url", "")
                    if image_data:
                        formatted_parts.append(build_image_part(image_data))
                # Pass through already provider-specific formats
                else:
                    formatted_parts.append(part)

            return {"role": message["role"], "content": formatted_parts}

        # Support explicitly interleaved parts via 'parts'
        parts = message.get('parts')
        if isinstance(parts, list):
            formatted_parts: List[Dict[str, Any]] = []

            for part in parts:
                if isinstance(part, str):
                    if part:
                        formatted_parts.append(build_text_part(part, message["role"]))
                    continue

                if not isinstance(part, dict):
                    logger.warning(f"Skipping unrecognized part format: {part}")
                    continue

                part_type = part.get("type")

                if part_type in ("text", "input_text", "output_text"):
                    text_value = part.get("text") or part.get("value")
                    if text_value:
                        formatted_parts.append(build_text_part(text_value, message["role"]))
                    continue

                if part_type == "image_url" and isinstance(part.get("image_url"), dict):
                    formatted_parts.append(part)
                    continue

                if part_type == "image":
                    image_input = part.get("image") or part.get("url") or part.get("path")
                    if not image_input:
                        logger.warning("Skipping image part without image input")
                        continue
                    detail = part.get("detail", "auto")
                    formatted_parts.append(build_image_part(image_input, detail))
                    continue

                # Allow raw OpenAI style dicts to pass through if they already match expected schema
                if part_type and part_type not in {"text", "image", "image_url"}:
                    formatted_parts.append(part)
                    continue

                logger.warning(f"Skipping unsupported part type: {part_type}")

            if formatted_parts:
                return {"role": message["role"], "content": formatted_parts}

        text = message.get('text')
        if text is None and isinstance(current_message_content, str):
            text = current_message_content
        
        # Support image_paths, direct base64 images, and URLs
        images = message.get('image_paths', message.get('images', []))
        image_details = message.get('image_details', ['auto'] * len(images))

        # Filter out unsupported image formats before processing
        if images:
            valid_images = []
            valid_details = []
            
            for img, detail in zip(images, image_details):
                try:
                    mime_type = get_mime_type(img)
                    if mime_type in self.SUPPORTED_MIME_TYPES:
                        valid_images.append(img)
                        valid_details.append(detail)
                    else:
                        logger.warning(f"Skipping unsupported image type: {mime_type} for image: {img}")
                except Exception as e:
                    logger.warning(f"Skipping image due to error determining format: {img} - {str(e)}")
            
            images = valid_images
            image_details = valid_details

        content: List[Dict[str, Any]] = []
        is_responses_api = isinstance(provider, OpenAIProvider)

        if text is not None:
            content.append(build_text_part(text, message["role"]))

        for img, detail in zip(images, image_details):
            try:
                content.append(build_image_part(img, detail))
            except Exception as e:
                logger.error(f"Error formatting image {img}: {str(e)}")
                raise ImageFormatError(f"Failed to format image {img}: {str(e)}")

        # Responses API requires content to always be a list
        if is_responses_api:
            return {"role": message["role"], "content": content or [build_text_part("", message["role"])]}
        # Other APIs can use string content when there are no images
        return {"role": message["role"], "content": content or text}

    def _get_provider(self, model_name: str) -> Union[UnifiedProvider, AnthropicProvider, BFLProvider, OpenAIProvider, GoogleGenAIProvider]:
        model_lower = model_name.lower()
        for provider_key, (provider_class, match_criteria) in self.PROVIDER_SETUP.items():
            for criterion in match_criteria:
                if model_lower.startswith(criterion):
                    if provider_key not in self._providers:
                        logger.info(f"Instantiating provider: {provider_key} with class {provider_class.__name__}")
                        if provider_class == UnifiedProvider:
                            self._providers[provider_key] = provider_class(provider_key, self.config)
                        else:
                            self._providers[provider_key] = provider_class(self.config)
                    return self._providers[provider_key]
        raise LLMError(f"Unsupported model: {model_name}")

    def stop_generation(self):
        if self._providers:
            for provider in self._providers.values():
                provider.stop_generation()