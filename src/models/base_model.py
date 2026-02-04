"""
Base Model Interface

Abstract base class for all AI model implementations.
Provides consistent interface for multi-model integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    GROQ = "groq"
    GEMINI = "gemini"


@dataclass
class ModelResponse:
    """Standardized response from any model."""
    content: str
    model_type: ModelType
    model_name: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None and self.content is not None


class BaseModel(ABC):
    """Abstract base class for AI models."""
    
    def __init__(self, api_key: str, model_name: str, model_type: ModelType):
        self.api_key = api_key
        self.model_name = model_name
        self.model_type = model_type
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: User prompt/query
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            ModelResponse with content or error
        """
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generate a structured JSON response.
        
        Args:
            prompt: User prompt/query
            schema: JSON schema for expected output
            system_prompt: Optional system instructions
            
        Returns:
            ModelResponse with JSON content
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
