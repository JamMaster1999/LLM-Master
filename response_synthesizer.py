import time
import base64
import logging
import requests
import os
from pathlib import Path
from typing import Dict, Optional, List, Union, Any, Generator

from .base_provider import UnifiedProvider, ProviderError
from .anthropic_provider import AnthropicProvider
from .classes import LLMResponse, LLMError, ModelRegistry
from .config import LLMConfig

logger = logging.getLogger(__name__)

class ImageFormatError(LLMError):
    """Raised when there's an issue with image formatting"""
    pass

class QueryLLM:
    """Main class for handling LLM interactions with retry logic and fallbacks"""

    SUPPORTED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
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
        logger.info("Initialized QueryLLM handler")

    def query(self,
              model_name: str,
              messages: List[Dict[str, Any]],
              stream: bool = False,
              fallback_provider: Optional[str] = None,
              fallback_model: Optional[str] = None,
              moderation: bool = False
    ) -> Union[LLMResponse, str]:
        """
        Generate a response (non-stream) or produce a generator (stream).

        If stream=True, we hand off to `_streaming_query` to yield chunks.
        Otherwise, we handle everything right here and return an LLMResponse/string.

        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries
            stream: Whether to stream the response
            fallback_provider: Optional fallback provider if primary fails
            fallback_model: Optional fallback model name
            moderation: Whether to perform content moderation

        Returns:
            - If stream=False: An LLMResponse (or string) with the full final output
            - If stream=True: A generator that yields partial chunks
        """
        if stream:
            # ---------------------------
            # STREAMING MODE:
            # Return a generator by calling `_streaming_query`
            # ---------------------------
            return self._streaming_query(
                model_name=model_name,
                messages=messages,
                fallback_provider=fallback_provider,
                fallback_model=fallback_model,
                moderation=moderation
            )
        else:
            # ---------------------------
            # NON-STREAMING MODE:
            # No yield => we return an actual object (LLMResponse/string)
            # ---------------------------
            start_time = time.time()
            retry_count = 0
            provider = self._get_provider(model_name)

            # Content moderation if enabled
            if moderation:
                self._moderate_content(messages)

            while retry_count <= self.MAX_RETRIES:
                try:
                    if retry_count > 0:
                        delay = 2 ** (retry_count - 1)
                        time.sleep(delay)
                        logger.info(f"Retry attempt {retry_count} after {delay}s delay...")

                    # Format messages for the provider
                    formatted_messages = [
                        self._format_message(msg, provider) for msg in messages
                    ]

                    # Call the provider with stream=False -> returns LLMResponse or string
                    response = provider.generate(
                        messages=formatted_messages,
                        model=model_name,
                        stream=False
                    )

                    # Add error checking for response type
                    if response is None:
                        raise LLMError("Provider returned None response")

                    # If the provider tracks usage details
                    if hasattr(provider, 'last_usage') and provider.last_usage is not None:
                        # Wrap final result in an LLMResponse that includes usage, cost, etc.
                        if isinstance(response, LLMResponse):
                            # The provider might already return an LLMResponse,
                            # but let's ensure we update the latency
                            response.latency = time.time() - start_time
                            return response
                        else:
                            # E.g. if `response` is raw text or something else:
                            from .classes import Usage
                            usage = provider.last_usage
                            final = LLMResponse(
                                content=str(response),
                                model_name=model_name,
                                usage=usage,
                                latency=time.time() - start_time
                            )
                            return final
                    else:
                        # If no usage info is available, we just return the raw response
                        return response

                except Exception as e:
                    # Get status code if available
                    status_code = getattr(e, 'status_code', None)
                    if status_code:
                        logger.error(f"Error {status_code}: {str(e)}")
                    else:
                        logger.error(f"Error: {str(e)}")

                    retry_count += 1
                    # Report only the first error
                    if retry_count == 1:
                        self._report_error(e)

                    # If out of retries but we have fallback
                    if retry_count > self.MAX_RETRIES and fallback_provider and fallback_model:
                        logger.info(f"Switching to fallback provider: {fallback_model}")
                        try:
                            fallback_response = self.query(
                                model_name=fallback_model,
                                messages=messages,
                                stream=False,
                                fallback_provider=None,
                                fallback_model=None,
                                moderation=False
                            )
                            return fallback_response
                        except Exception as fallback_e:
                            self._report_error(fallback_e)
                            raise LLMError("Both primary and fallback providers failed")

                    elif retry_count > self.MAX_RETRIES:
                        status = getattr(e, 'status_code', '')
                        status_str = f" ({status})" if status else ""
                        raise LLMError(f"Max retries exceeded. Error{status_str}: {str(e)}")

                finally:
                    logger.info(f"Total time: {time.time() - start_time:.2f}s")

    def _streaming_query(self,
                         model_name: str,
                         messages: List[Dict[str, Any]],
                         fallback_provider: Optional[str] = None,
                         fallback_model: Optional[str] = None,
                         moderation: bool = False
    ) -> Generator[str, None, None]:
        """
        Internal method to handle streaming. Returns a generator of chunks.
        """
        start_time = time.time()
        retry_count = 0
        provider = self._get_provider(model_name)

        if moderation:
            self._moderate_content(messages)

        while retry_count <= self.MAX_RETRIES:
            try:
                if retry_count > 0:
                    delay = 2 ** (retry_count - 1)
                    time.sleep(delay)
                    logger.info(f"Retry attempt {retry_count} after {delay}s delay...")

                # Format messages for the provider
                formatted_messages = [
                    self._format_message(msg, provider) for msg in messages
                ]

                # Use the provider in streaming mode
                full_response = ""
                stream_gen = provider.generate(
                    messages=formatted_messages,
                    model=model_name,
                    stream=True
                )

                for chunk in stream_gen:
                    full_response += chunk
                    yield chunk  # <--- The actual streaming output to the caller

                # Post-stream, handle usage if available
                if hasattr(provider, 'last_usage') and provider.last_usage is not None:
                    # You could log or store usage here if you want
                    pass

                break  # If we got this far, it succeeded—stop retrying

            except Exception as e:
                retry_count += 1
                # Report only the first error
                if retry_count == 1:
                    self._report_error(e)

                if retry_count > self.MAX_RETRIES and fallback_provider and fallback_model:
                    logger.info(f"Switching to fallback provider (stream): {fallback_model}")
                    # Use `yield from` so we can stream fallback chunks
                    yield from self._streaming_query(
                        model_name=fallback_model,
                        messages=messages,
                        fallback_provider=None,
                        fallback_model=None,
                        moderation=False  # Already moderated
                    )
                    return

                elif retry_count > self.MAX_RETRIES:
                    raise LLMError(f"Max retries exceeded (stream). Last error: {str(e)}")

            finally:
                logger.info(f"Total time (stream): {time.time() - start_time:.2f}s")

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

        def encode_image(image_path: str) -> str:
            path_obj = Path(image_path)
            if not path_obj.exists():
                raise ImageFormatError(f"Image file not found: {path_obj}")
            with open(path_obj, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # If already has 'content', assume it's well-structured
        if 'content' in message:
            return message

        text = message.get('text')
        image_paths = message.get('image_paths', [])

        if not image_paths:
            return {"role": message["role"], "content": text}

        content = []
        if text:
            content.append({"type": "text", "text": text})

        for path in image_paths:
            try:
                base64_image = encode_image(path)
                mime_type = f"image/{Path(path).suffix.lower()[1:]}"
                if mime_type == "image/jpg":
                    mime_type = "image/jpeg"
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
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
            except Exception as e:
                logger.error(f"Error formatting image {path}: {str(e)}")
                raise ImageFormatError(f"Failed to format image {path}: {str(e)}")

        return {"role": message["role"], "content": content}

    def _get_provider(self, model_name: str) -> Union[UnifiedProvider, AnthropicProvider]:
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
            "claude": ("anthropic", AnthropicProvider),
            "gemini": ("gemini", UnifiedProvider),
            "mistral": ("mistral", UnifiedProvider)
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