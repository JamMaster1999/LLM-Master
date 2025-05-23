from typing import Dict, Optional
from dataclasses import dataclass
import os

@dataclass
class LLMConfig:
    """Configuration class for LLM API keys and settings"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    recraft_api_key: Optional[str] = None
    fireworks_api_key: Optional[str] = None
    bfl_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            recraft_api_key=os.getenv("RECRAFT_API_KEY"),
            fireworks_api_key=os.getenv("FIREWORKS_API_KEY"),
            bfl_api_key=os.getenv("BFL_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY")
        )
    
    def validate(self) -> Dict[str, bool]:
        """Validate which providers are configured"""
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "gemini": bool(self.gemini_api_key),
            "mistral": bool(self.mistral_api_key),
            "recraft": bool(self.recraft_api_key),
            "fireworks": bool(self.fireworks_api_key),
            "bfl": bool(self.bfl_api_key),
            "perplexity": bool(self.perplexity_api_key)
        }
