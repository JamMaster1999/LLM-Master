from typing import List, Dict, Any, Union, Generator, Optional, AsyncGenerator
import asyncio
import os
from posthog import Posthog
from posthog.ai.openai import OpenAI
try:
    from openai import AsyncOpenAI, DefaultAioHttpClient
except ImportError:  # pragma: no cover - fallback for older SDK versions
    AsyncOpenAI = None
    DefaultAioHttpClient = None
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

class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models"""
    supports_native_async = True
    API_KEY_ATTR = "openai_api_key"
    SUPPORTS_CACHING = True

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        api_key = getattr(self.config, self.API_KEY_ATTR, None)
        self._native_client = None
        self._native_http_client = None
            
        try:
            self.posthog = _init_posthog()
            self.client = OpenAI(api_key=api_key, posthog_client=self.posthog)
            self.supports_caching = self.SUPPORTS_CACHING
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            logger.info("Successfully initialized OpenAI provider")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize OpenAI provider: {str(e)}")

    def _get_or_create_native_client(self):
        if AsyncOpenAI is None:
            raise ConfigurationError("AsyncOpenAI is unavailable. Please upgrade the openai package to use native async clients.")
        if self._native_client is None:
            api_key = getattr(self.config, self.API_KEY_ATTR, None) or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError("Missing OpenAI API key for async client")
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if DefaultAioHttpClient is not None:
                self._native_http_client = DefaultAioHttpClient()
                client_kwargs["http_client"] = self._native_http_client
            self._native_client = AsyncOpenAI(**client_kwargs)
        return self._native_client

    async def _create_sync_response(self, request_params: Dict[str, Any]) -> Any:
        return await asyncio.get_event_loop().run_in_executor(None, lambda: self.client.responses.create(**request_params))

    async def _create_native_response(self, request_params: Dict[str, Any]) -> Any:
        return await self._get_or_create_native_client().responses.create(**request_params)

    @staticmethod
    def _strip_model_prefix(model: str) -> str:
        return model[len("responses-"):] if model.startswith("responses-") else model

    @staticmethod
    def _prepare_request(messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        local_kwargs = dict(kwargs)
        model = local_kwargs.pop("model", None)
        if not model:
            raise ValueError("Model parameter is required")
        if "max_tokens" in local_kwargs and "max_output_tokens" not in local_kwargs:
            local_kwargs["max_output_tokens"] = local_kwargs.pop("max_tokens")
        if "max_output_tokens" not in local_kwargs:
            local_kwargs["max_output_tokens"] = 1024
        posthog_params = local_kwargs.pop("posthog", None)
        request_params = {"model": OpenAIProvider._strip_model_prefix(model), "input": messages, **local_kwargs}
        return model, request_params, posthog_params if isinstance(posthog_params, dict) else None

    async def aclose(self) -> None:
        if self._native_client is not None:
            await self._native_client.aclose()
            self._native_client = None

        if self._native_http_client is not None and hasattr(self._native_http_client, "aclose"):
            await self._native_http_client.aclose()
            self._native_http_client = None

    @staticmethod
    def _apply_posthog_params(request_params: Dict[str, Any], posthog_params: Optional[Dict[str, Any]]) -> None:
        if not posthog_params:
            return
        for key, value in posthog_params.items():
            request_params[f"posthog_{key}"] = value

    @staticmethod
    def _raise_on_error(response: Any) -> None:
        if getattr(response, "error", None):
            raise ProviderError(f"OpenAI API error: {response.error.code} - {response.error.message}")

    @staticmethod
    def _extract_text(response: Any) -> str:
        segments: List[str] = []
        if getattr(response, "output", None):
            for item in response.output:
                if item.type == "message" and item.role == "assistant" and item.content:
                    for block in item.content:
                        if block.type == "output_text":
                            segments.append(block.text)
        if not segments and isinstance(getattr(response, "output_text", None), str):
            segments.append(response.output_text)
        return "".join(segments)

    @staticmethod
    def _build_usage(response: Any) -> Usage:
        usage_meta = getattr(response, "usage", None)
        if not usage_meta:
            return Usage(input_tokens=0, output_tokens=0, cached_tokens=0)
        cached_tokens = 0
        details = getattr(usage_meta, "prompt_tokens_details", None)
        if details:
            cached_tokens = getattr(details, "cached_tokens", 0)
        return Usage(
            input_tokens=getattr(usage_meta, "input_tokens", 0) or 0,
            output_tokens=getattr(usage_meta, "output_tokens", 0) or 0,
            cached_tokens=cached_tokens,
        )

    async def generate(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        kwargs = dict(kwargs)
        if stream:
            return self._stream_response(messages, **kwargs)

        use_native_async = kwargs.pop("_native_async", False)

        model, request_params, posthog_params = self._prepare_request(messages, kwargs)
        logger.debug("Generating response with model=%s native_async=%s", model, use_native_async)

        if use_native_async:
            response = await self._create_native_response(request_params)
        else:
            self._apply_posthog_params(request_params, posthog_params)
            response = await self._create_sync_response(request_params)

        self._raise_on_error(response)

        content = self._extract_text(response)
        if not content:
            logger.warning("No output text found in response (%s)", model)

        usage = self._build_usage(response)
        self.last_usage = usage
        self.last_response = content

        return LLMResponse(content=content, model_name=model, usage=usage, latency=0.0)

    async def _stream_response(self, 
                        messages: List[Dict[str, Any]], 
                        **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream a response from the OpenAI API (async generator)
        
        Args:
            messages: List of message dictionaries (passed as 'input' to API)
            **kwargs: Additional arguments to pass to the API
        
        Yields:
            Chunks of the generated response
        """
        model = kwargs.pop('model', None)
        if not model:
            raise ValueError("Model parameter is required")

        # Map max_tokens to max_output_tokens
        if 'max_tokens' in kwargs and 'max_output_tokens' not in kwargs:
            kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
        # Ensure max_output_tokens has a default value
        if 'max_output_tokens' not in kwargs:
            kwargs['max_output_tokens'] = 1024
        
        # Extract PostHog parameters if provided as a dict
        posthog_params = kwargs.pop('posthog', None)
        if posthog_params and isinstance(posthog_params, dict):
            # Unpack PostHog parameters with posthog_ prefix
            for key, value in posthog_params.items():
                kwargs[f'posthog_{key}'] = value

        logger.debug(f"Starting streaming response with model: {model}")
        
        full_response = ""
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        stream = None
        in_reasoning_block = False
        in_code_block = False
        web_search_queries = {}  # Store queries by search ID
        web_search_results = {}  # Store results by search ID
        final_citations = []  # Store all citations for final list
        
        api_model_name = self._strip_model_prefix(model)
        request_params = {"model": api_model_name, "input": messages, "stream": True, **kwargs}

        try:
            stream = self.client.responses.create(**request_params)
            self._current_generation = stream 

            for event in stream:
                # Process events based on type
                if event.type == 'response.reasoning_summary_text.delta':
                    reasoning_delta = event.delta
                    # Open the block if it's the first reasoning delta
                    if not in_reasoning_block:
                        yield "\n\n<think>\n\n"
                        in_reasoning_block = True
                    yield reasoning_delta

                elif event.type == 'response.output_text.delta':
                    # Close the reasoning block *before* the first output text delta
                    if in_reasoning_block:
                        yield "\n\n</think>\n\n"
                        in_reasoning_block = False # Reasoning is definitely over
                    
                    # Close the code block *before* the first output text delta
                    if in_code_block:
                        yield "\n```\n\n"
                        in_code_block = False
                    
                    # Yield the actual output text delta
                    delta = event.delta
                    full_response += delta
                    yield delta
                
                elif event.type == 'response.code_interpreter_call.in_progress':
                    # Close reasoning block if open when code starts
                    # if in_reasoning_block:
                    #     yield "\n\n</think>\n\n"
                    #     in_reasoning_block = False
                    
                    # Start code block
                    if not in_code_block:
                        yield "\n```py\n"
                        in_code_block = True
                
                elif event.type == 'response.code_interpreter_call_code.delta':
                    # Ensure we're in a code block (should already be from in_progress event)
                    if not in_code_block:
                        yield "\n```py\n"
                        in_code_block = True
                    
                    # Yield the code delta
                    code_delta = event.delta
                    yield code_delta
                
                elif event.type == 'response.code_interpreter_call.completed':
                    # Close code block when interpretation is complete
                    if in_code_block:
                        yield "\n```\n\n"
                        in_code_block = False
                
                elif event.type == 'response.output_item.done':
                    # Only process if event.item exists
                    if hasattr(event, 'item') and event.item is not None:
                        # Handle web search completion
                        if getattr(event.item, 'type', None) == 'web_search_call':
                            try:
                                query = event.item.to_dict().get('action', {}).get('query', 'Unknown')
                                if query != 'Unknown':
                                    yield f"\n**Search Query**: {query}\n"
                                yield "✅ Search completed\n"
                            except:
                                yield "✅ Search completed\n"
                        
                        # Check if this is a code interpreter call with outputs
                        elif (hasattr(event.item, 'type') and 
                            event.item.type == 'code_interpreter_call' and
                            hasattr(event.item, 'outputs') and 
                            event.item.outputs):
                            
                            # Collect all logs first to check if we have any content
                            logs_content = []
                            images = []
                            
                            for output in event.item.outputs:
                                # Handle both dict and object formats
                                output_type = getattr(output, 'type', None) or output.get('type') if isinstance(output, dict) else None
                                
                                if output_type == 'logs':
                                    logs = getattr(output, 'logs', None) or output.get('logs') if isinstance(output, dict) else None
                                    if logs and logs.strip():  # Only add non-empty logs
                                        logs_content.append(logs)
                                elif output_type == 'image':
                                    url = getattr(output, 'url', None) or output.get('url') if isinstance(output, dict) else None
                                    if url:
                                        images.append(url)
                            
                            # Only display logs block if we have actual content
                            if logs_content:
                                yield "\n```plaintext\n"
                                for logs in logs_content:
                                    yield logs
                                yield "\n```\n\n"
                            
                            # Display images
                            for url in images:
                                yield f"<code_execution_image>{url}</code_execution_image>\n\n"
                        
                        # Handle annotations (file citations and URL citations)
                        if hasattr(event.item, 'content') and event.item.content:
                            for content_item in event.item.content:
                                if hasattr(content_item, 'annotations') and content_item.annotations:
                                    for annotation in content_item.annotations:
                                        if annotation.type == 'container_file_citation':
                                            yield f"\n\n<code_execution_file><code_execution_file_id>{annotation.file_id}</code_execution_file_id><code_execution_file_name>{annotation.filename}</code_execution_file_name><code_execution_container_id>{annotation.container_id}</code_execution_container_id></code_execution_file>\n\n"
                                        elif annotation.type == 'url_citation' and annotation.url and annotation.title:
                                            citation_entry = {'url': annotation.url, 'title': annotation.title}
                                            if citation_entry not in final_citations:
                                                final_citations.append(citation_entry)
                
                elif event.type == 'response.web_search_call.in_progress':
                    yield "\n🔍 **Starting Web Search**\n"
                
                elif event.type == 'response.completed':
                    # Close reasoning block if it was still open when completion event arrives
                    if in_reasoning_block:
                         yield "\n\n</think>\n\n"
                         in_reasoning_block = False
                    
                    # Close code block if it was still open when completion event arrives
                    if in_code_block:
                         yield "\n```\n\n"
                         in_code_block = False
                    
                    # Process usage etc.
                    if hasattr(event, 'usage'):
                        usage = event.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        
                        # Access cached_tokens from prompt_tokens_details as per OpenAI docs
                        cached_tokens = 0
                        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                            cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                        
                        self.last_usage = Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_tokens=cached_tokens
                        )
                        logger.debug(f"Completed streaming with {input_tokens} input, {output_tokens} output, {cached_tokens} cached tokens")
                    break # Exit loop on completion
                
                elif event.type == 'error':
                    # Close reasoning block if open on error
                    if in_reasoning_block:
                         yield "\n\n</think>\n\n"
                         in_reasoning_block = False
                    
                    # Close code block if open on error
                    if in_code_block:
                         yield "\n```\n\n"
                         in_code_block = False
                    
                    logger.error(f"OpenAI streaming error: {event.code} - {event.message}")
                    raise ProviderError(f"OpenAI streaming error: {event.code} - {event.message}")

                # Note: We are IGNORING other event types like response.reasoning_summary_part.added/done
                # for the purpose of tag placement. We only care about the transition from reasoning deltas
                # to output text deltas.
            
            # Store the full response after successful streaming
            self.last_response = full_response
                    
        except Exception as e:
            # Close reasoning block if an exception occurs during the loop
            if in_reasoning_block:
                 # Attempt to yield closing tag, might fail if connection broken
                 try: yield "\n\n</think>\n\n" 
                 except: pass 
                 in_reasoning_block = False
            
            # Close code block if an exception occurs during the loop
            if in_code_block:
                 # Attempt to yield closing tag, might fail if connection broken
                 try: yield "\n```\n\n" 
                 except: pass 
                 in_code_block = False
            
            logger.error(f"Error during streaming: {str(e)}")
            if isinstance(e, ProviderError):
                 raise
            raise ProviderError(f"OpenAI streaming failed: {str(e)}")
            
        finally:
            # Final check: Ensure reasoning block is closed if stream ends for any reason
            if in_reasoning_block:
                 # Attempt to yield closing tag
                 try: yield "\n\n</think>\n\n" 
                 except: pass 
            
            # Final check: Ensure code block is closed if stream ends for any reason
            if in_code_block:
                 # Attempt to yield closing tag
                 try: yield "\n```\n\n" 
                 except: pass 
            
            # Output final citation list if not already done
            if final_citations:
                try:
                    yield "\n<citation_list>\n"
                    for citation in final_citations:
                        yield f"<citation_source><citation_url>{citation['url']}</citation_url></citation_source>\n"
                    yield "</citation_list>\n"
                except: pass
            
            self._current_generation = None

    def stop_generation(self):
        """Stop the current generation if any"""
        if self._current_generation:
            logger.info("Attempting to stop current OpenAI generation")
            # Similar to Anthropic, just clearing the reference might be enough
            # if iteration stops. If the library offers a specific method, call it here.
            # e.g., if self._current_generation has a close() method:
            # try:
            #     self._current_generation.close()
            # except Exception as e:
            #     logger.warning(f"Could not explicitly close stream: {e}")
            self._current_generation = None
            logger.info("OpenAI generation reference cleared.")
    