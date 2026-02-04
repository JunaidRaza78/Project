"""
Model Manager

Manages multiple AI models with intelligent routing and fallback.
"""

import asyncio
from typing import Any, Optional
from enum import Enum

from .base_model import BaseModel, ModelResponse, ModelType
from .groq_model import GroqModel


class TaskType(Enum):
    """Types of tasks for model routing."""
    FAST_EXTRACTION = "fast_extraction"  # Use faster model
    COMPLEX_REASONING = "complex_reasoning"  # Use larger model
    STRUCTURED_OUTPUT = "structured_output"  # Either, prefer faster


class ModelManager:
    """
    Manages multiple AI models with routing and fallback.
    
    Routing Strategy (Groq-only mode):
    - Fast extraction tasks -> llama-3.1-8b-instant (faster)
    - Complex reasoning -> llama-3.3-70b-versatile (better reasoning)
    - Fallback between models on errors
    """
    
    def __init__(
        self,
        groq_api_key: str,
        google_api_key: str = "",  # Optional, not used in Groq-only mode
        groq_model: str = "llama-3.3-70b-versatile",
        groq_fast_model: str = "llama-3.1-8b-instant",
        gemini_model: str = "",  # Not used
    ):
        # Primary model for complex reasoning
        self.groq = GroqModel(groq_api_key, groq_model)
        # Fast model for extraction tasks
        self.groq_fast = GroqModel(groq_api_key, groq_fast_model)
        
        # Alias for backward compatibility
        self.gemini = self.groq  # Use Groq as fallback too
        
        # Track rate limiting
        self._groq_rate_limited = False
        self._rate_limit_reset_time = 0
    
    def _get_model_for_task(self, task_type: TaskType) -> tuple[BaseModel, BaseModel]:
        """Get primary and fallback model for task type."""
        if task_type == TaskType.COMPLEX_REASONING:
            # Use large model, fallback to fast
            return self.groq, self.groq_fast
        else:
            # Use fast model for speed, fallback to large
            return self.groq_fast, self.groq
    
    async def generate(
        self,
        prompt: str,
        task_type: TaskType = TaskType.FAST_EXTRACTION,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """
        Generate response with automatic model selection and fallback.
        """
        primary, fallback = self._get_model_for_task(task_type)
        
        # Try primary model
        response = await primary.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if response.success:
            return response
        
        # Check if rate limited
        if "rate" in response.error.lower():
            self._groq_rate_limited = True
        
        # Try fallback model
        fallback_response = await fallback.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return fallback_response
    
    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        task_type: TaskType = TaskType.STRUCTURED_OUTPUT,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Generate structured JSON with fallback."""
        primary, fallback = self._get_model_for_task(task_type)
        
        response = await primary.generate_structured(
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
        )
        
        if response.success:
            return response
        
        # Try fallback
        return await fallback.generate_structured(
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
        )
    
    async def parallel_generate(
        self,
        prompts: list[str],
        task_type: TaskType = TaskType.FAST_EXTRACTION,
        system_prompt: Optional[str] = None,
    ) -> list[ModelResponse]:
        """Generate multiple prompts in parallel."""
        tasks = [
            self.generate(
                prompt=prompt,
                task_type=task_type,
                system_prompt=system_prompt,
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
