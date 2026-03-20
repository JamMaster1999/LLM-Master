from typing import List, Optional, Union, Dict, Any, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from enum import Enum
import logging
import asyncio
import re
import uuid

logger = logging.getLogger(__name__)

# Try to import modal for distributed rate limiting; fall back to local mode if unavailable
try:
    import modal
    MODAL_AVAILABLE = True
    lock_queue = modal.Queue.from_name("rate_limit_lock_queue", create_if_missing=True)
    if lock_queue.len() == 0:
        lock_queue.put("LOCK_TOKEN", block=False)
except Exception:
    MODAL_AVAILABLE = False
    lock_queue = None
    logger.info("Modal not installed - using local rate limiting (no distributed coordination)")

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
        # "gpt-4o-mini": ModelConfig(0.15, 0.60, 0.075, 10000),
        # "gpt-4o": ModelConfig(2.50, 10.00, 1.25, 10000),
        # "o4-mini": ModelConfig(1.1, 4.4, 0.275, 10000),
        # "o3": ModelConfig(2.00, 8.00, 0.5, 10000),
        "omni-moderation-latest": ModelConfig(0.00, 0.00, None, 1000),
        "gpt-4o-mini-audio-preview": ModelConfig(0.15, 0.60, 0.075, 5000),
        "gpt-4o-audio-preview": ModelConfig(2.50, 10.00, 1.25, 5000),
        "chatgpt-4o-latest": ModelConfig(5.00, 15.00, None, 5000),
        "gpt-4.1": ModelConfig(2.00, 8.00, 0.5, 10000),
        "gpt-5-chat-latest": ModelConfig(1.25, 10.00, 0.125, 10000),
        "gpt-image-1": ModelConfig(5.00, 40.00, None, 50),
        
        
        # OpenAI Response Models
        "responses-gpt-4o": ModelConfig(2.50, 10.00, 1.25, 10000),
        "responses-gpt-4.1": ModelConfig(2.00, 8.00, 0.5, 10000),
        "responses-o4-mini": ModelConfig(1.1, 4.4, 0.275, 10000),
        "responses-o3": ModelConfig(2.00, 8.00, 0.5, 10000),
        "responses-gpt-5": ModelConfig(1.25, 10.00, 0.125, 10000),
        "responses-gpt-5.1": ModelConfig(1.25, 10.00, 0.125, 10000),
        "responses-gpt-5.2": ModelConfig(1.75, 14.00, 0.175, 10000),
        "responses-gpt-5.4": ModelConfig(2.5, 15.00, 0.25, 10000),
        "responses-gpt-5-codex": ModelConfig(1.25, 10.00, 0.125, 10000),
        "responses-gpt-5-mini": ModelConfig(0.25, 2.00, 0.025, 10000),

        # Anthropic Models
        # "claude-3-5-sonnet-latest": ModelConfig(3.00, 15.00, 0.30, 4000),
        # "claude-3-7-sonnet-latest": ModelConfig(3.00, 15.00, 0.30, 4000),
        # "claude-sonnet-4-20250514": ModelConfig(3.00, 15.00, 0.30, 4000),
        "claude-haiku-4-5-20251001": ModelConfig(1.00, 5.00, 0.10, 4000),
        "claude-sonnet-4-5-20250929": ModelConfig(3.00, 15.00, 0.30, 4000),
        "claude-sonnet-4-6": ModelConfig(3.00, 15.00, 0.30, 4000),
        "claude-opus-4-5-20251101": ModelConfig(5.00, 25.00, 0.50, 4000),
        "claude-opus-4-6": ModelConfig(5.00, 25.00, 0.50, 4000),

        # Gemini Models
        "googleai:gemini-2.5-flash-lite": ModelConfig(0.1, 0.4, 0.01, 30000),
        "googleai:gemini-2.5-flash-lite-preview-09-2025": ModelConfig(0.1, 0.4, 0.01, 30000),
        "googleai:gemini-2.5-flash": ModelConfig(0.3, 2.5, 0.03, 10000),
        "googleai:gemini-2.5-flash-preview-09-2025": ModelConfig(0.3, 2.5, 0.075, 10000),
        "googleai:gemini-2.5-pro": ModelConfig(1.25, 10, 0.125, 2000),
        "googleai:gemini-3-pro-preview": ModelConfig(2, 12, 0.2, 2000),
        "googleai:gemini-3.1-pro-preview": ModelConfig(2, 12, 0.2, 2000),
        "googleai:gemini-3-flash-preview": ModelConfig(0.5, 3, 0.05, 20000),
        "googleai:gemini-2.5-flash-image": ModelConfig(0.3, 0.039, None, 5000),
        "googleai:gemini-3-pro-image": ModelConfig(2, 0.134, None, 2000),

        # Vertex Models
        "vertexai:gemini-2.5-flash-lite": ModelConfig(0.1, 0.4, 0.01, 30000),
        "vertexai:gemini-2.5-flash-lite-preview-09-2025": ModelConfig(0.1, 0.4, 0.01, 30000),
        "vertexai:gemini-2.5-flash": ModelConfig(0.3, 2.5, 0.03, 10000),
        "vertexai:gemini-2.5-flash-preview-09-2025": ModelConfig(0.3, 2.5, 0.075, 10000),
        "vertexai:gemini-2.5-pro": ModelConfig(1.25, 10, 0.125, 2000),
        "vertexai:gemini-3-pro-preview": ModelConfig(2, 12, 0.2, 2000),
        "vertexai:gemini-3.1-pro-preview": ModelConfig(2, 12, 0.2, 2000),
        "vertexai:gemini-3-flash-preview": ModelConfig(0.5, 3, 0.05, 20000),
        "vertexai:gemini-2.5-flash-image": ModelConfig(0.3, 0.039, None, 5000),
        "vertexai:gemini-3-pro-image": ModelConfig(2, 0.134, None, 2000),


        # Recraft Models
        "recraftv3": ModelConfig(0.00, 0.04, None, 100),

        # Fireworks Models
        "accounts/fireworks/models/deepseek-r1-0528": ModelConfig(3, 8, None, 600),
        
        # BFL Models
        # "flux-dev": ModelConfig(0.00, 0.025, None, 24),
        # "flux-pro-1.1": ModelConfig(0.00, 0.04, None, 24),
        
        # Perplexity Models
        "sonar": ModelConfig(1, 1, None, 50),
        "sonar-pro": ModelConfig(3, 15, None, 50),
        # "sonar-reasoning": ModelConfig(1, 5, None, 50),
        # "sonar-reasoning-pro": ModelConfig(2, 8, None, 50),
        # "sonar-deep-research": ModelConfig(2, 8, None, 5),
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
    supports_native_async: bool = False
    
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
    """
    Rate limiter with two modes:
    - Modal mode: Distributed coordination via Modal Queue/Dict (when modal is installed)
    - Local mode: In-memory rate limiting (fallback when modal is not available)
    """
    def __init__(self, model_config: ModelConfig, model_name: str):
        self.model_config = model_config
        self.model_name = model_name
        self.use_modal = MODAL_AVAILABLE
        
        sanitized_name = self._sanitize_name(model_name)
        
        if self.use_modal:
            self.request_queue = modal.Queue.from_name(f"llm_queue_{sanitized_name}", create_if_missing=True)
            self.request_dict = modal.Dict.from_name(f"llm_requests_{sanitized_name}", create_if_missing=True)
            self.response_dict = modal.Dict.from_name(f"llm_responses_{sanitized_name}", create_if_missing=True)
            self.rate_dict = modal.Dict.from_name(f"llm_rate_limits_{sanitized_name}", create_if_missing=True)
            if "request_timestamps" not in self.rate_dict:
                self.rate_dict["request_timestamps"] = []
        else:
            # Local in-memory storage
            self._local_timestamps: List[float] = []
            self._local_token_usage: int = 0
            self._local_requests: Dict[str, dict] = {}
            self._local_responses: Dict[str, Any] = {}
            self._local_queue: asyncio.Queue = asyncio.Queue()
            self._local_lock = asyncio.Lock()
    
    def _sanitize_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    
    def _prune_old_timestamps(self, timestamps: List[float]) -> List[float]:
        now = time.time()
        return [ts for ts in timestamps if now - ts < 60]
    
    def _reserve_capacity(self, timestamps: List[float], batch_size: int) -> List[float]:
        if batch_size <= 0:
            batch_size = 1
        now = time.time()
        increment = 1e-6
        timestamps.extend(now + (i * increment) for i in range(batch_size))
        return timestamps
    
    def _has_capacity(self, timestamps: List[float], batch_size: int) -> bool:
        if batch_size <= 0:
            batch_size = 1
        return len(timestamps) + batch_size <= self.model_config.rate_limit_rpm
        
    async def wait_for_capacity(self, batch_size: int = 1):
        """Block until capacity is available for batch_size requests."""
        first_wait = True
        
        if self.use_modal:
            while True:
                await lock_queue.get.aio()
                try:
                    timestamps = self._prune_old_timestamps(self.rate_dict["request_timestamps"])
                    if self._has_capacity(timestamps, batch_size):
                        timestamps = self._reserve_capacity(timestamps, batch_size)
                        self.rate_dict["request_timestamps"] = timestamps
                        if not first_wait:
                            logger.info(f"Capacity available for {self.model_name}, resuming processing")
                        return
                    else:
                        if first_wait:
                            logger.info(
                                f"Rate limit reached for {self.model_name} "
                                f"({len(timestamps)}/{self.model_config.rate_limit_rpm}). Waiting for capacity..."
                            )
                            first_wait = False
                finally:
                    await lock_queue.put.aio("LOCK_TOKEN")
                await asyncio.sleep(1)
        else:
            # Local mode
            while True:
                async with self._local_lock:
                    self._local_timestamps = self._prune_old_timestamps(self._local_timestamps)
                    if self._has_capacity(self._local_timestamps, batch_size):
                        self._local_timestamps = self._reserve_capacity(self._local_timestamps, batch_size)
                        if not first_wait:
                            logger.info(f"Capacity available for {self.model_name}, resuming processing")
                        return
                    else:
                        if first_wait:
                            logger.info(
                                f"Rate limit reached for {self.model_name} "
                                f"({len(self._local_timestamps)}/{self.model_config.rate_limit_rpm}). Waiting for capacity..."
                            )
                            first_wait = False
                await asyncio.sleep(1)

    async def can_make_request(self, batch_size: int = 1) -> bool:
        """Check and reserve capacity without blocking; returns True if successful."""
        if self.use_modal:
            await lock_queue.get.aio()
            try:
                timestamps = self._prune_old_timestamps(self.rate_dict["request_timestamps"])
                if self._has_capacity(timestamps, batch_size):
                    timestamps = self._reserve_capacity(timestamps, batch_size)
                    self.rate_dict["request_timestamps"] = timestamps
                    return True
                else:
                    self.rate_dict["request_timestamps"] = timestamps
                    return False
            finally:
                await lock_queue.put.aio("LOCK_TOKEN")
        else:
            async with self._local_lock:
                self._local_timestamps = self._prune_old_timestamps(self._local_timestamps)
                if self._has_capacity(self._local_timestamps, batch_size):
                    self._local_timestamps = self._reserve_capacity(self._local_timestamps, batch_size)
                    return True
                return False

    async def add_token_usage(self, tokens: int):
        """Track token usage."""
        if self.use_modal:
            if "token_usage" not in self.rate_dict:
                self.rate_dict["token_usage"] = 0
            current_usage = self.rate_dict["token_usage"]
            self.rate_dict["token_usage"] = current_usage + tokens
        else:
            self._local_token_usage += tokens
            current_usage = self._local_token_usage - tokens
        logger.info(f"Added {tokens} tokens to {self.model_name}. Total usage: {current_usage + tokens}")
        
    async def submit_request(self, request: dict) -> str:
        """Submit request and store for processing."""
        request_id = str(uuid.uuid4())
        
        if self.use_modal:
            self.request_dict[request_id] = request
            await self.request_queue.put.aio(request_id)
        else:
            self._local_requests[request_id] = request
            await self._local_queue.put(request_id)
        
        logger.debug(f"Submitted request {request_id[:8]} to {self.model_name}")
        return request_id
        
    async def wait_for_response(self, request_id: str, timeout: int = 60):
        """Wait for specific request ID's response."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.use_modal:
                if request_id in self.response_dict:
                    response = self.response_dict[request_id]
                    del self.response_dict[request_id]
                    if request_id in self.request_dict:
                        del self.request_dict[request_id]
                    logger.debug(f"Got response for request {request_id[:8]}")
                    return response
            else:
                if request_id in self._local_responses:
                    response = self._local_responses.pop(request_id)
                    self._local_requests.pop(request_id, None)
                    logger.debug(f"Got response for request {request_id[:8]}")
                    return response
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")
        
    async def process_queue(self):
        """Process queue (runs in background)."""
        while True:
            if await self.can_make_request():
                try:
                    if self.use_modal:
                        request_id = await self.request_queue.get(timeout=1)
                        if request_id in self.request_dict:
                            request = self.request_dict[request_id]
                            logger.debug(f"Processing request {request_id[:8]}")
                            return request_id, request
                    else:
                        try:
                            request_id = await asyncio.wait_for(self._local_queue.get(), timeout=1)
                            if request_id in self._local_requests:
                                request = self._local_requests[request_id]
                                logger.debug(f"Processing request {request_id[:8]}")
                                return request_id, request
                        except asyncio.TimeoutError:
                            pass
                except TimeoutError:
                    pass
            await asyncio.sleep(0.1)
        
    def store_response(self, request_id: str, response: Any):
        """Store response for a request."""
        if self.use_modal:
            self.response_dict[request_id] = response
        else:
            self._local_responses[request_id] = response
        logger.info(f"Stored response for request {request_id[:8]} in {self.model_name}")