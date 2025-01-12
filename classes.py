from typing import List, Optional, Union, Dict, Any, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from enum import Enum
import logging
import modal
import asyncio

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass

class ModelNotFoundError(LLMError):
    """Raised when a requested model is not found in the registry"""
    pass

class PricingError(LLMError):
    """Raised when there's an error calculating costs"""
    pass

@dataclass
class ModelConfig:
    """Configuration for each model including pricing and rate limits
    
    Attributes:
        input_price_per_million: Cost per million input tokens
        output_price_per_million: Cost per million output tokens
        cached_input_price_per_million: Cost per million cached input tokens (if applicable)
        rate_limit_rpm: Rate limit in requests per minute
        supports_streaming: Whether the model supports streaming
    """
    input_price_per_million: float
    output_price_per_million: float
    cached_input_price_per_million: Optional[float] = None
    rate_limit_rpm: int = 0
    supports_streaming: bool = True

class ModelRegistry:
    """Registry of all supported models and their configurations"""
    
    CONFIGS = {
        # OpenAI Models
        "gpt-4o-mini": ModelConfig(0.15, 0.60, 0.075, 5000),
        "gpt-4o": ModelConfig(2.50, 10.00, 1.25, 5000),
        "omni-moderation-latest": ModelConfig(0.00, 0.00, None, 1000),
        # Anthropic Models
        "claude-3-5-sonnet-latest": ModelConfig(3.00, 15.00, 0.30, 4000),
        "claude-3-5-haiku-latest": ModelConfig(1.00, 5.00, 0.10, 4000),
        # Gemini Models
        "gemini-1.5-pro-latest": ModelConfig(1.25, 5.00, None, 1000),
        "gemini-1.5-flash-latest": ModelConfig(0.075, 0.30, None, 2000),
        "gemini-2.0-flash-exp": ModelConfig(0.075, 0.30, None, 10),
        "gemini-2.0-flash-thinking-exp": ModelConfig(0.075, 0.30, None, 10),
        # Mistral Models
        "mistral-large-latest": ModelConfig(2.00, 6.00, None, 300)
    }

    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a specific model
        
        Args:
            model_name: Name of the model to look up
            
        Returns:
            ModelConfig for the specified model
            
        Raises:
            ModelNotFoundError: If the model is not found in the registry
        """
        try:
            return cls.CONFIGS[model_name]
        except KeyError:
            logger.error(f"Model not found in registry: {model_name}")
            raise ModelNotFoundError(f"Model '{model_name}' not found in registry")

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available models"""
        return list(cls.CONFIGS.keys())

@dataclass
class Usage:
    """Track token usage and calculate costs
    
    Attributes:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        cached_tokens: Number of cached tokens used (if applicable)
    """
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    
    def calculate_cost(self, model_config: ModelConfig) -> float:
        """Calculate the total cost based on token usage
        
        Args:
            model_config: ModelConfig instance containing pricing information
            
        Returns:
            Total cost in USD
            
        Raises:
            PricingError: If there's an error calculating the cost
        """
        try:
            input_cost = (self.input_tokens / 1_000_000) * model_config.input_price_per_million
            output_cost = (self.output_tokens / 1_000_000) * model_config.output_price_per_million
            cached_cost = 0
            
            if model_config.cached_input_price_per_million and self.cached_tokens > 0:
                cached_cost = (self.cached_tokens / 1_000_000) * model_config.cached_input_price_per_million
            
            return input_cost + output_cost + cached_cost
            
        except Exception as e:
            logger.error(f"Error calculating cost: {str(e)}")
            raise PricingError(f"Failed to calculate cost: {str(e)}")

class LLMResponse:
    """Standardized response object for LLM interactions
    
    Attributes:
        content: The generated text content
        model_name: Name of the model used
        usage: Token usage information
        latency: Response time in seconds
        cost: Calculated cost of the request
    """
    def __init__(self, 
                 content: str,
                 model_name: str,
                 usage: Usage,
                 latency: float):
        self.content = content
        self.model_name = model_name
        self.usage = usage
        self.latency = latency
        
        try:
            model_config = ModelRegistry.get_config(model_name)
            self.cost = usage.calculate_cost(model_config)
        except (ModelNotFoundError, PricingError) as e:
            logger.error(f"Error setting response cost: {str(e)}")
            self.cost = 0.0

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers
    
    All LLM providers must implement these methods to ensure
    consistent behavior across different providers.
    """
    
    @abstractmethod
    async def generate(self, 
                messages: List[Dict[str, Any]], 
                model: str,
                stream: bool = False,
                **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        """Generate a response from the LLM
        
        Args:
            messages: List of message dictionaries
            model: Name of the model to use
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Either a LLMResponse object or a Generator for streaming
        """
        pass

    @abstractmethod
    def stop_generation(self):
        """Stop the current generation if any"""
        pass

class ResponseSynthesizer:
    """Main class for handling LLM interactions"""
    def __init__(self):
        self._providers = {}

    def generate_response(self,
                         model_name: str,
                         messages: List[Dict[str, Any]],
                         stream: bool = False) -> Union[LLMResponse, Generator[str, None, None]]:
        start_time = time.time()
        
        try:
            provider = self._get_provider(model_name)
            response = provider.generate(messages, stream=stream)
            
            if not stream:
                response.latency = time.time() - start_time
            
            return response
            
        except Exception as e:
            # Handle errors and potentially implement fallback logic
            raise

    def stop_generation(self):
        """Stop the current generation if any"""
        if self._providers:
            # Stop the actual provider's generation
            for provider in self._providers.values():
                provider.stop_generation()

    def _get_provider(self, model_name: str) -> BaseLLMProvider:
        """Get or create appropriate provider for the model"""
        provider_map = {
            "gpt": "OpenAIProvider",
            "claude": "AnthropicProvider",
            "gemini": "GeminiProvider",
            "mistral": "MistralProvider"
        }
        
        # Determine provider from model name
        provider_key = next(
            (key for key in provider_map if key in model_name.lower()),
            None
        )
        
        if not provider_key:
            raise ValueError(f"Unsupported model: {model_name}")
            
        if provider_key not in self._providers:
            # Initialize provider (we'll implement these classes next)
            provider_class = globals()[provider_map[provider_key]]
            self._providers[provider_key] = provider_class()
            
        return self._providers[provider_key]

class RateLimiter:
    """Distributed rate limiter using Modal Queue and Dict for request-response handling"""
    def __init__(self, model_config: ModelConfig, model_name: str):
        self.model_config = model_config
        self.model_name = model_name
        
        # Queue only stores request IDs for coordination
        self.request_queue = modal.Queue.from_name(f"llm_queue_{model_name}", create_if_missing=True)
        
        # Dict stores actual request/response data
        self.request_dict = modal.Dict.from_name(f"llm_requests_{model_name}", create_if_missing=True)
        self.response_dict = modal.Dict.from_name(f"llm_responses_{model_name}", create_if_missing=True)
        
        # Dict for rate limiting
        self.rate_dict = modal.Dict.from_name(f"llm_rate_limits_{model_name}", create_if_missing=True)
        
        # Initialize rate tracking
        if "request_timestamps" not in self.rate_dict:
            self.rate_dict["request_timestamps"] = []
            
    def _cleanup_old_requests(self):
        """Remove requests older than 1 minute"""
        import time
        current_time = time.time()
        timestamps = self.rate_dict["request_timestamps"]
        self.rate_dict["request_timestamps"] = [ts for ts in timestamps if current_time - ts < 60]
        
    async def can_make_request(self) -> bool:
        """Check if we can make a request based on rate limits"""
        self._cleanup_old_requests()
        timestamps = self.rate_dict["request_timestamps"]
        return len(timestamps) < self.model_config.rate_limit_rpm
        
    async def wait_for_capacity(self):
        """Wait until there is capacity to make a request"""
        while not await self.can_make_request():
            await asyncio.sleep(1)  # Wait a second before checking again
            
    async def submit_request(self, request: dict) -> str:
        """Submit request and store in Dict"""
        import uuid
        request_id = str(uuid.uuid4())
        
        # Store actual request data in Dict
        self.request_dict[request_id] = request
        
        # Only queue the request ID
        await self.request_queue.put(request_id)
        
        # Track this request
        timestamps = self.rate_dict["request_timestamps"]
        timestamps.append(time.time())
        self.rate_dict["request_timestamps"] = timestamps
        
        return request_id
        
    async def wait_for_response(self, request_id: str, timeout: int = 60):
        """Wait for specific request ID's response"""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.response_dict:
                response = self.response_dict[request_id]
                del self.response_dict[request_id]  # Cleanup
                if request_id in self.request_dict:
                    del self.request_dict[request_id]  # Cleanup request too
                return response
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")
        
    async def process_queue(self):
        """Process queue (runs in background)"""
        while True:
            if await self.can_make_request():
                try:
                    # Get only the request ID from queue
                    request_id = await self.request_queue.get(timeout=1)
                    
                    # Get actual request data from Dict
                    if request_id in self.request_dict:
                        request = self.request_dict[request_id]
                        return request_id, request
                    
                except TimeoutError:
                    pass
            await asyncio.sleep(0.1)
        
    def store_response(self, request_id: str, response: Any):
        """Store response for a request"""
        self.response_dict[request_id] = response
        
    async def add_token_usage(self, tokens: int):
        """Track token usage"""
        if "token_usage" not in self.rate_dict:
            self.rate_dict["token_usage"] = 0
        current_usage = self.rate_dict["token_usage"]
        self.rate_dict["token_usage"] = current_usage + tokens