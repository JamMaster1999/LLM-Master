from typing import List, Dict, Any, Union, Generator, Optional, AsyncGenerator
import asyncio
import os
from anthropic import Anthropic
try:
    from anthropic import AsyncAnthropic, DefaultAioHttpClient
except ImportError:  # pragma: no cover - fallback for older SDK versions
    AsyncAnthropic = None
    DefaultAioHttpClient = None
from posthog import Posthog
from posthog.ai.anthropic import Anthropic as PostHogAnthropic
from .classes import BaseLLMProvider, LLMResponse, Usage
from .config import LLMConfig
import logging
from .base_provider import ProviderError, ConfigurationError

logger = logging.getLogger(__name__)

def _init_posthog() -> Posthog:
    return Posthog(
        project_api_key=os.getenv("POSTHOG_API_KEY", "phc_1uBDKATKfxK7ougGiL9F9hnCgeXJvc4k6TMP2oekfnK"),
        host=os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
    )

class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic models"""
    supports_native_async = True
    API_KEY_ATTR = "anthropic_api_key"
    SUPPORTS_CACHING = True

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        api_key = getattr(self.config, self.API_KEY_ATTR)
        if not api_key:
            raise ConfigurationError("Missing Anthropic API key")
        self._native_client = None
        self._native_http_client = None
            
        try:
            self.posthog = _init_posthog()
            self.client = PostHogAnthropic(api_key=api_key, posthog_client=self.posthog)
            self.supports_caching = self.SUPPORTS_CACHING
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            logger.info("Successfully initialized Anthropic provider")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize Anthropic provider: {str(e)}")

    def _get_or_create_native_client(self):
        if AsyncAnthropic is None:
            raise ConfigurationError("AsyncAnthropic is unavailable. Please upgrade the anthropic package to use native async clients.")
        if self._native_client is None:
            api_key = getattr(self.config, self.API_KEY_ATTR, None) or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ConfigurationError("Missing Anthropic API key for async client")
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if DefaultAioHttpClient is not None:
                self._native_http_client = DefaultAioHttpClient()
                client_kwargs["http_client"] = self._native_http_client
            self._native_client = AsyncAnthropic(**client_kwargs)
        return self._native_client

    async def _create_sync_response(self, model: str, payload_messages: List[Dict[str, Any]], request_kwargs: Dict[str, Any]) -> Any:
        return await asyncio.get_event_loop().run_in_executor(None, lambda: self.client.messages.create(model=model, messages=payload_messages, **request_kwargs))

    async def _create_native_response(self, model: str, payload_messages: List[Dict[str, Any]], request_kwargs: Dict[str, Any]) -> Any:
        return await self._get_or_create_native_client().messages.create(model=model, messages=payload_messages, **request_kwargs)

    @staticmethod
    def _prepare_request(messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]]]:
        local_kwargs = dict(kwargs)
        working_messages = [dict(msg) for msg in messages]
        system_blocks: List[Dict[str, Any]] = []
        cleaned_messages: List[Dict[str, Any]] = []

        for msg in working_messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list):
                    system_blocks.extend(content)
                else:
                    block = {"type": "text", "text": content}
                    if "cache_control" in msg:
                        block["cache_control"] = msg["cache_control"]
                    system_blocks.append(block)
            else:
                cleaned_messages.append(msg)

        if system_blocks:
            local_kwargs["system"] = system_blocks

        model = local_kwargs.pop("model", None)
        if not model:
            raise ValueError("Model parameter is required")

        if "max_tokens" not in local_kwargs:
            local_kwargs["max_tokens"] = 1024

        posthog_params = local_kwargs.pop("posthog", None)
        return model, cleaned_messages, local_kwargs, posthog_params if isinstance(posthog_params, dict) else None

    async def aclose(self) -> None:
        if self._native_client is not None:
            await self._native_client.aclose()
            self._native_client = None

        if self._native_http_client is not None and hasattr(self._native_http_client, "aclose"):
            await self._native_http_client.aclose()
            self._native_http_client = None

    @staticmethod
    def _apply_posthog_params(request_kwargs: Dict[str, Any], posthog_params: Optional[Dict[str, Any]]) -> None:
        if not posthog_params:
            return
        for key, value in posthog_params.items():
            request_kwargs[f"posthog_{key}"] = value

    @staticmethod
    def _raise_on_error(response: Any) -> None:
        if getattr(response, "error", None):
            raise ProviderError(f"Anthropic API error: {response.error}")

    @staticmethod
    def _extract_text(response: Any) -> str:
        if not getattr(response, "content", None):
            return ""
        return "".join(chunk.text for chunk in response.content if getattr(chunk, "type", None) == "text")

    @staticmethod
    def _build_usage(response: Any) -> Usage:
        usage_meta = getattr(response, "usage", None)
        if not usage_meta:
            return Usage(input_tokens=0, output_tokens=0, cached_tokens=0)
        return Usage(
            input_tokens=getattr(usage_meta, "input_tokens", 0) or 0,
            output_tokens=getattr(usage_meta, "output_tokens", 0) or 0,
            cached_tokens=getattr(usage_meta, "cache_read_input_tokens", 0) or 0,
        )

    async def generate(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        kwargs = dict(kwargs)
        use_native_async = kwargs.pop("_native_async", False)

        model, payload_messages, request_kwargs, posthog_params = self._prepare_request(messages, kwargs)

        if stream:
            self._apply_posthog_params(request_kwargs, posthog_params)
            return self._stream_response(payload_messages, model=model, **request_kwargs)

        logger.debug("Generating response with model=%s native_async=%s", model, use_native_async)

        if use_native_async:
            response = await self._create_native_response(model, payload_messages, request_kwargs)
        else:
            self._apply_posthog_params(request_kwargs, posthog_params)
            response = await self._create_sync_response(model, payload_messages, request_kwargs)

        self._raise_on_error(response)

        content = self._extract_text(response)
        usage = self._build_usage(response)
        self.last_usage = usage
        self.last_response = content

        return LLMResponse(content=content, model_name=model, usage=usage, latency=0.0)

    def _stream_response(self, 
                        messages: List[Dict[str, Any]], 
                        **kwargs) -> Generator[str, None, None]:
        """
        Stream a response from the Anthropic API
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments to pass to the API
        
        Yields:
            Chunks of the generated response
        """
        try:
            model = kwargs.pop('model', None)
            if not model:
                raise ValueError("Model parameter is required")
                
            if 'max_tokens' not in kwargs:
                kwargs['max_tokens'] = 1024
            
            posthog_params = kwargs.pop('posthog', None)
            self._apply_posthog_params(kwargs, posthog_params)
                
            logger.debug(f"Starting streaming response with model: {model}")
            
            full_response = ""
            input_tokens = 0
            output_tokens = 0
            cached_tokens = 0
            
            # Use create with stream=True for PostHog compatibility
            kwargs['stream'] = True
            stream = self.client.messages.create(
                model=model,
                messages=messages,
                **kwargs
            )
            self._current_generation = stream
            
            for event in stream:
                # Get input tokens from MessageStartEvent
                if event.type == 'message_start' and hasattr(event.message, 'usage'):
                    input_tokens = event.message.usage.input_tokens
                    cached_tokens = getattr(event.message.usage, 'cache_read_input_tokens', 0)
                
                # Get output tokens from MessageDeltaEvent
                elif event.type == 'message_delta' and hasattr(event, 'usage'):
                    output_tokens = event.usage.output_tokens
                
                # Handle content streaming
                elif event.type == 'content_block_delta':
                    content = event.delta.text
                    full_response += content
                    yield content
            
            # Store usage data
            self.last_usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens
            )
            
            # Store the full response
            self.last_response = full_response
            
            logger.debug(f"Completed streaming with {input_tokens} input and {output_tokens} output tokens")
                    
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise ProviderError(f"Anthropic streaming failed: {str(e)}")
            
        finally:
            self._current_generation = None

    def stop_generation(self):
        """Stop the current generation if any"""
        if self._current_generation:
            logger.info("Stopping current generation")
            # Anthropic's stream uses context manager, 
            # so we just need to clear the reference
            self._current_generation = None
