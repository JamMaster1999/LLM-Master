from typing import List, Dict, Any, Union, Generator, Optional, AsyncGenerator
import asyncio
from openai import OpenAI  # Changed import
from .classes import BaseLLMProvider, LLMResponse, Usage
from .config import LLMConfig
import logging
from .base_provider import ProviderError, ConfigurationError

# Set up logging
logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models"""
    
    PROVIDER_CONFIG = {
        "client_class": OpenAI,
        "api_key_attr": "openai_api_key",  # Assuming this exists in LLMConfig
        "supports_caching": True # Based on usage details provided
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the OpenAI provider
        
        Args:
            config: LLMConfig instance. If None, will attempt to load from environment
        """
        self.config = config or LLMConfig.from_env()
        
        # Get API key from config
        # The OpenAI client can also load from OPENAI_API_KEY env var if key is None
        api_key = getattr(self.config, self.PROVIDER_CONFIG["api_key_attr"], None)
            
        try:
            # OpenAI client defaults to reading OPENAI_API_KEY from env if api_key is not passed
            self.client = self.PROVIDER_CONFIG["client_class"](api_key=api_key)
            self.supports_caching = self.PROVIDER_CONFIG["supports_caching"]
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            
            logger.info("Successfully initialized OpenAI provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize OpenAI provider: {str(e)}")

    async def generate(self, 
                messages: List[Dict[str, Any]], 
                stream: bool = False,
                **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        """
        Generate a response using the OpenAI API
        
        Args:
            messages: List of message dictionaries (passed as 'input' to API)
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to the API
        
        Returns:
            Either a LLMResponse object or a Generator for streaming
        """
        try:
            if stream:
                # Streaming needs to be an async generator
                return self._stream_response(messages, **kwargs) 

            # Extract model from kwargs
            model = kwargs.pop('model', None)
            if not model:
                raise ValueError("Model parameter is required")
            
            # Map max_tokens to max_output_tokens if provided
            if 'max_tokens' in kwargs and 'max_output_tokens' not in kwargs:
                 kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
            # Ensure max_output_tokens has a default value if not provided
            if 'max_output_tokens' not in kwargs:
                kwargs['max_output_tokens'] = 1024 # Default like Anthropic's max_tokens
            
            logger.debug(f"Generating response with model: {model}")

            # Strip prefix for API call
            api_model_name = model
            if api_model_name.startswith("responses-"):
                api_model_name = api_model_name[len("responses-"):]
                logger.debug(f"Stripped prefix, using API model name: {api_model_name}")

            request_params = {
                "model": api_model_name, # API uses 'input' and expects stripped model name
                "input": messages, 
                **kwargs
            }
            
            # Assuming the client might be synchronous based on examples
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.responses.create(**request_params)
            )

            # Check for errors in the response object
            if response.error:
                logger.error(f"OpenAI API error: {response.error.code} - {response.error.message}")
                raise ProviderError(f"OpenAI API error: {response.error.code} - {response.error.message}")

            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_tokens=getattr(response.usage.input_tokens_details, 'cached_tokens', 0) # Get cached tokens if available
            )

            # Extract text content - need to search the output list
            content = ""
            if response.output:
                for output_item in response.output:
                    if output_item.type == "message" and output_item.role == "assistant":
                        if output_item.content:
                            for content_item in output_item.content:
                                if content_item.type == "output_text":
                                    content += content_item.text
                                    # Assuming we only care about the first text block if multiple exist
                                    # or concatenate if needed? Sticking to concatenation for now.
            
            if not content:
                 logger.warning("No output text found in response.")
                 # Check if response.output_text convenience attribute exists and has text
                 if hasattr(response, 'output_text') and isinstance(response.output_text, str):
                     content = response.output_text # Use convenience attribute if primary search fails

            logger.debug(f"Generated response with {usage.input_tokens} input, {usage.output_tokens} output, {usage.cached_tokens} cached tokens")

            return LLMResponse(
                content=content,
                model_name=model, # Return the original model name with prefix
                usage=usage,
                latency=0  # Will be set by ResponseSynthesizer
            )
            
        except Exception as e:
            # Catch specific OpenAI errors if known, otherwise general exception
            logger.error(f"Error during generation: {str(e)}")
            # Check if it's already a ProviderError or ConfigurationError
            if isinstance(e, (ProviderError, ConfigurationError)):
                raise
            raise ProviderError(f"OpenAI generation failed: {str(e)}")

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

        logger.debug(f"Starting streaming response with model: {model}")
        
        full_response = ""
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        stream = None # Define stream variable outside try
        in_reasoning_block = False # State variable
        
        # Strip prefix for API call
        api_model_name = model
        if api_model_name.startswith("responses-"):
            api_model_name = api_model_name[len("responses-"):]
            logger.debug(f"Stripped prefix, using API model name: {api_model_name}")

        request_params = {
            "model": api_model_name, # Use stripped name
            "input": messages,
            "stream": True,
            **kwargs
        }

        try:
            # Assuming client returns an iterable stream, possibly sync iteration
            # Needs adaptation if client provides an async iterator directly
            # For now, wrap sync iteration in async generator structure
            
            stream = self.client.responses.create(**request_params)
            self._current_generation = stream 

            # How to make this async? If stream is sync iterable:
            # We might need run_in_executor or async thread handling if iteration blocks.
            # Or if the SDK has an async version, use that.
            # Let's assume for now the user will adapt this part if needed,
            # or that iterating the stream object might work okay within asyncio.
            # A truly async implementation would use `async for event in stream:`
            # if the client library supports it.

            for event in stream:
                # Process events based on type
                if event.type == 'response.reasoning_summary_text.delta':
                    reasoning_delta = event.delta
                    # Open the block if it's the first reasoning delta
                    if not in_reasoning_block:
                        yield "<think>"
                        in_reasoning_block = True
                    yield reasoning_delta

                elif event.type == 'response.output_text.delta':
                    # Close the reasoning block *before* the first output text delta
                    if in_reasoning_block:
                        yield "</think>\n\n"
                        in_reasoning_block = False # Reasoning is definitely over
                    
                    # Yield the actual output text delta
                    delta = event.delta
                    full_response += delta
                    yield delta
                
                elif event.type == 'response.completed':
                    # Close reasoning block if it was still open when completion event arrives
                    if in_reasoning_block:
                         yield "</think>\n\n"
                         in_reasoning_block = False
                    
                    # Process usage etc.
                    if hasattr(event, 'usage'):
                        usage = event.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        if hasattr(usage, 'input_tokens_details'):
                           cached_tokens = getattr(usage.input_tokens_details, 'cached_tokens', 0)
                        
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
                         yield "</think>\n\n"
                         in_reasoning_block = False
                         
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
                 try: yield "</think>\n\n" 
                 except: pass 
                 in_reasoning_block = False
                 
            logger.error(f"Error during streaming: {str(e)}")
            if isinstance(e, ProviderError):
                 raise
            raise ProviderError(f"OpenAI streaming failed: {str(e)}")
            
        finally:
            # Final check: Ensure reasoning block is closed if stream ends for any reason
            if in_reasoning_block:
                 # Attempt to yield closing tag
                 try: yield "</think>\n\n" 
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