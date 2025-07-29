from typing import List, Optional, Union, Dict, Any, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from enum import Enum
import logging
import modal
import asyncio
import re

logger = logging.getLogger(__name__)

# Initialize the lock queue for rate limiting
lock_queue = modal.Queue.from_name("rate_limit_lock_queue", create_if_missing=True)

# Ensure exactly one lock token exists
if lock_queue.len() == 0:
    lock_queue.put("LOCK_TOKEN", block=False)  # Non-blocking put for initialization

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
        "chatgpt-4o-latest": ModelConfig(5.00, 15.00, None, 5000),
        "gpt-4o-mini-audio-preview": ModelConfig(0.15, 0.60, 0.075, 5000),
        "gpt-4o-audio-preview": ModelConfig(2.50, 10.00, 1.25, 5000),
        "o3-mini": ModelConfig(1.1, 4.4, 0.275, 5000),
        "o4-mini": ModelConfig(1.1, 4.4, 0.275, 5000),
        "o3": ModelConfig(2.00, 8.00, 0.5, 5000),
        "gpt-4.1": ModelConfig(2.00, 8.00, 0.5, 5000),
        "gpt-4.1-mini": ModelConfig(0.4, 1.6, 0.1, 5000),
        "gpt-image-1": ModelConfig(5.00, 40.00, None, 50),
        "omni-moderation-latest": ModelConfig(0.00, 0.00, None, 1000),
        
        # OpenAI Response Models
        "responses-gpt-4o": ModelConfig(2.50, 10.00, 1.25, 5000),
        "responses-o4-mini": ModelConfig(1.1, 4.4, 0.275, 5000),
        "responses-o3": ModelConfig(2.00, 8.00, 0.5, 5000),
        "responses-gpt-4.1": ModelConfig(2.00, 8.00, 0.5, 5000),
        "responses-gpt-4.1-mini": ModelConfig(0.4, 1.6, 0.1, 5000),

        # Anthropic Models
        "claude-3-5-sonnet-latest": ModelConfig(3.00, 15.00, 0.30, 4000),
        "claude-3-7-sonnet-latest": ModelConfig(3.00, 15.00, 0.30, 4000),
        "claude-sonnet-4-20250514": ModelConfig(3.00, 15.00, 0.30, 4000),

        # Gemini Models
        "gemini-2.5-flash-lite": ModelConfig(0.1, 0.4, 0.025, 30000),
        "gemini-2.5-flash": ModelConfig(0.3, 2.5, 0.075, 10000),
        "gemini-2.5-pro": ModelConfig(1.25, 10, 0.31, 2000),
        "imagen-3.0-generate-002": ModelConfig(0.00, 0.03, None, 20),

        # Recraft Models
        "recraftv3": ModelConfig(0.00, 0.04, None, 100),

        # Fireworks Models
        "accounts/fireworks/models/deepseek-r1-0528": ModelConfig(3, 8, None, 600),
        
        # BFL Models
        "flux-dev": ModelConfig(0.00, 0.025, None, 24),
        "flux-pro-1.1": ModelConfig(0.00, 0.04, None, 24),
        
        # Perplexity Models
        "sonar": ModelConfig(1, 1, None, 50),
        "sonar-pro": ModelConfig(3, 15, None, 50),
        "sonar-reasoning": ModelConfig(1, 5, None, 50),
        "sonar-reasoning-pro": ModelConfig(2, 8, None, 50),
        "sonar-deep-research": ModelConfig(2, 8, None, 5),
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
            # Calculate cost for non-cached input tokens
            non_cached_input_tokens = self.input_tokens - self.cached_tokens
            input_cost = (non_cached_input_tokens / 1_000_000) * model_config.input_price_per_million
            
            # Calculate cost for output tokens
            output_cost = (self.output_tokens / 1_000_000) * model_config.output_price_per_million

            # Calculate cost for cached input tokens
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
        audio_data: Optional base64-encoded audio data (for audio-capable models)
        citations: Optional list of citations (for Perplexity models)
    """
    def __init__(self, 
                 content: str,
                 model_name: str,
                 usage: Usage,
                 latency: float,
                 audio_data: Optional[str] = None,
                 citations: Optional[List[str]] = None):
        self.content = content
        self.model_name = model_name
        self.usage = usage
        self.latency = latency
        self.audio_data = audio_data
        self.citations = citations
        
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

class RateLimiter:
    """Distributed rate limiter using a lock queue plus timestamp-based checking."""
    def __init__(self, model_config: ModelConfig, model_name: str):
        self.model_config = model_config
        self.model_name = model_name
        
        # Sanitize model name for queue names (replace slashes and other invalid chars with underscores)
        sanitized_name = self._sanitize_name(model_name)
        
        # Queue only stores request IDs for coordination
        self.request_queue = modal.Queue.from_name(f"llm_queue_{sanitized_name}", create_if_missing=True)
        
        # Dict stores actual request/response data
        self.request_dict = modal.Dict.from_name(f"llm_requests_{sanitized_name}", create_if_missing=True)
        self.response_dict = modal.Dict.from_name(f"llm_responses_{sanitized_name}", create_if_missing=True)
        
        # Dict for rate limiting
        self.rate_dict = modal.Dict.from_name(f"llm_rate_limits_{sanitized_name}", create_if_missing=True)
        
        # Initialize rate tracking
        if "request_timestamps" not in self.rate_dict:
            self.rate_dict["request_timestamps"] = []
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize model name for use in Modal Queue and Dict names.
        Replace invalid characters with underscores.
        """
        # Replace slashes, spaces and other potentially problematic characters
        return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
        
    async def wait_for_capacity(self):
        """
        Acquire our distributed 'lock_queue' to ensure 
        read/check/write is atomic. We'll loop until capacity is found.
        """
        import time
        
        first_wait = True
        while True:
            # 1) Acquire the lock
            await lock_queue.get.aio()
            try:
                # 2) Read timestamps
                timestamps = self.rate_dict["request_timestamps"]
                
                # 3) Prune old
                now = time.time()
                timestamps = [ts for ts in timestamps if now - ts < 60]
                
                # 4) Check capacity
                if len(timestamps) < self.model_config.rate_limit_rpm:
                    # We have capacity => append
                    timestamps.append(now)
                    self.rate_dict["request_timestamps"] = timestamps

                    # Log if we had to wait
                    if not first_wait:
                        logger.info(f"Capacity available for {self.model_name}, resuming processing")
                    
                    return  # Done
                else:
                    # No capacity => must wait
                    if first_wait:
                        logger.info(
                            f"Rate limit reached for {self.model_name} "
                            f"({len(timestamps)}/{self.model_config.rate_limit_rpm}). Waiting for capacity..."
                        )
                        first_wait = False
            finally:
                # 5) Release the lock so other tasks can check
                await lock_queue.put.aio("LOCK_TOKEN")
            
            # Wait a second before re-checking
            await asyncio.sleep(1)

    async def can_make_request(self) -> bool:
        """
        Do the same logic, but return True/False immediately 
        instead of blocking until capacity.
        """
        import time
        
        # Acquire the lock
        await lock_queue.get.aio()
        try:
            timestamps = self.rate_dict["request_timestamps"]
            now = time.time()
            timestamps = [ts for ts in timestamps if now - ts < 60]
            
            if len(timestamps) < self.model_config.rate_limit_rpm:
                # Append now
                timestamps.append(now)
                self.rate_dict["request_timestamps"] = timestamps
                return True
            else:
                # Update pruned timestamps even when at capacity
                self.rate_dict["request_timestamps"] = timestamps
                return False
        finally:
            await lock_queue.put.aio("LOCK_TOKEN")

    async def add_token_usage(self, tokens: int):
        """Track token usage"""
        if "token_usage" not in self.rate_dict:
            self.rate_dict["token_usage"] = 0
        current_usage = self.rate_dict["token_usage"]
        self.rate_dict["token_usage"] = current_usage + tokens
        logger.info(f"Added {tokens} tokens to {self.model_name}. Total usage: {current_usage + tokens}")
        
    async def submit_request(self, request: dict) -> str:
        """Submit request and store in Dict"""
        import uuid
        request_id = str(uuid.uuid4())
        
        # Store actual request data in Dict
        self.request_dict[request_id] = request
        
        # Only queue the request ID
        await self.request_queue.put.aio(request_id)
        
        logger.debug(f"Submitted request {request_id[:8]} to {self.model_name}")
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
                logger.debug(f"Got response for request {request_id[:8]}")
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
                        logger.debug(f"Processing request {request_id[:8]}")
                        return request_id, request
                    
                except TimeoutError:
                    pass
            await asyncio.sleep(0.1)
        
    def store_response(self, request_id: str, response: Any):
        """Store response for a request"""
        self.response_dict[request_id] = response
        logger.info(f"Stored response for request {request_id[:8]} in {self.model_name}")