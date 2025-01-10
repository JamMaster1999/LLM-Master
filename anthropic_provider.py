from typing import List, Dict, Any, Union, Generator, Optional
from anthropic import Anthropic
from .classes import BaseLLMProvider, LLMResponse, Usage
from .config import LLMConfig
import logging
from .base_provider import ProviderError, ConfigurationError

# Set up logging
logger = logging.getLogger(__name__)

class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic models"""
    
    PROVIDER_CONFIG = {
        "client_class": Anthropic,
        "api_key_attr": "anthropic_api_key",
        "supports_caching": True  # Anthropic supports caching but we're not implementing it yet
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the Anthropic provider
        
        Args:
            config: LLMConfig instance. If None, will attempt to load from environment
        """
        self.config = config or LLMConfig.from_env()
        
        # Get API key from config
        api_key = getattr(self.config, self.PROVIDER_CONFIG["api_key_attr"])
        if not api_key:
            raise ConfigurationError("Missing Anthropic API key")
            
        try:
            self.client = self.PROVIDER_CONFIG["client_class"](api_key=api_key)
            self.supports_caching = self.PROVIDER_CONFIG["supports_caching"]
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            
            logger.info("Successfully initialized Anthropic provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize Anthropic provider: {str(e)}")

    def generate(self, 
                messages: List[Dict[str, Any]], 
                stream: bool = False,
                **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        """
        Generate a response using the Anthropic API
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to the API
        
        Returns:
            Either a LLMResponse object or a Generator for streaming
        """
        try:
            if stream:
                return self._stream_response(messages, **kwargs)

            # Extract model from kwargs to avoid duplicate parameter
            model = kwargs.pop('model', None)
            if not model:
                raise ValueError("Model parameter is required")
            
            logger.debug(f"Generating response with model: {model}")
            
            response = self.client.messages.create(
                model=model,
                max_tokens=kwargs.get('max_tokens', 1024),
                messages=messages,
                **kwargs
            )

            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_tokens=0  # Not implementing caching yet
            )

            # Anthropic returns content as array of ContentBlock objects
            content = "".join(
                chunk.text for chunk in response.content 
                if chunk.type == "text"
            )

            logger.debug(f"Generated response with {usage.input_tokens} input and {usage.output_tokens} output tokens")

            return LLMResponse(
                content=content,
                model_name=model,
                usage=usage,
                latency=0  # Will be set by ResponseSynthesizer
            )
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise ProviderError(f"Anthropic generation failed: {str(e)}")

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
                
            logger.debug(f"Starting streaming response with model: {model}")
            
            full_response = ""
            input_tokens = 0
            output_tokens = 0
            
            with self.client.messages.stream(
                model=model,
                max_tokens=kwargs.get('max_tokens', 1024),
                messages=messages,
                **kwargs
            ) as stream:
                self._current_generation = stream
                
                for event in stream:
                    # Get input tokens from MessageStartEvent
                    if event.type == 'message_start' and hasattr(event.message, 'usage'):
                        input_tokens = event.message.usage.input_tokens
                    
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
                    cached_tokens=0
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