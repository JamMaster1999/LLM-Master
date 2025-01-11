from typing import List, Dict, Any, Union, Generator, Optional
from openai import OpenAI
from mistralai import Mistral
import google.generativeai as genai
from .classes import BaseLLMProvider, LLMResponse, Usage
from .config import LLMConfig
import logging
import os

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
            "supports_caching": True
        },
        "gemini": {
            "client_class": OpenAI,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key_attr": "gemini_api_key",
            "supports_caching": False
        },
        "mistral": {
            "client_class": Mistral,
            "base_url": None,
            "api_key_attr": "mistral_api_key",
            "supports_caching": False
        }
    }

    def __init__(self, provider: str = "openai", config: Optional[LLMConfig] = None):
        """
        Initialize the provider with configuration
        
        Args:
            provider: The provider to use ("openai", "gemini", or "mistral")
            config: LLMConfig instance. If None, will attempt to load from environment
        """
        self.provider = provider
        self.config = config or LLMConfig.from_env()
        
        provider_config = self.PROVIDER_CONFIGS.get(provider)
        if not provider_config:
            raise ConfigurationError(f"Unsupported provider: {provider}")
            
        # Get API key from config
        api_key = getattr(self.config, provider_config["api_key_attr"])
        if not api_key:
            raise ConfigurationError(f"Missing API key for provider: {provider}")
            
        try:
            if provider == "gemini":
                genai.configure(api_key=api_key)
                
            if provider in ["openai", "gemini"]:
                self.client = provider_config["client_class"](
                    base_url=provider_config["base_url"],
                    api_key=api_key
                )
            else:  # mistral
                self.client = provider_config["client_class"](
                    api_key=api_key
                )
                
            self.supports_caching = provider_config["supports_caching"]
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            
            logger.info(f"Successfully initialized {provider} provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize {provider} provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize {provider} provider: {str(e)}")

    def _count_gemini_tokens(self, messages: List[Dict[str, Any]], model: str, gemini_model=None) -> int:
        """Helper function to count tokens for Gemini, including images"""
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

    def generate(self, 
            messages: List[Dict[str, Any]], 
            stream: bool = False,
            **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        if stream:
            return self._stream_response(messages, **kwargs)
        
        if self.provider in ["openai", "gemini"]:
            model = kwargs.pop('model', None)
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs
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
                input_tokens = self._count_gemini_tokens(messages, model, gemini_model)
                
                # Count output tokens
                output_count = gemini_model.count_tokens(response.choices[0].message.content)
                
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
        
            
        else:  # mistral
            model = kwargs.pop('model', None)
            response = self.client.chat.complete(
                model=model,
                messages=messages,
                **kwargs
            )
            
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached_tokens=0
            )
            
            content = response.choices[0].message.content
        
        return LLMResponse(
            content=content,
            model_name=model,
            usage=usage,
            latency=0  # Will be set by ResponseSynthesizer
        )

    def _stream_response(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        try:
            if self.provider in ["openai", "gemini"]:
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
                    stream_options={"include_usage": True} if self.provider == "openai" else None,
                    **kwargs
                )
                
                self._current_generation = stream
                
                for chunk in stream:
                    if self.provider == "openai":
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
            if self.provider in ["openai", "gemini"]:
                self._current_generation.close()
            self._current_generation = None