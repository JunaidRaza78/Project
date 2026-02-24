"""
Model Manager

Manages multiple AI models with intelligent routing and fallback.
Gemini 2.5 Flash for complex reasoning, Groq for fast extraction.
"""

import asyncio
from typing import Any, Optional
from enum import Enum

from .base_model import BaseModel, ModelResponse, ModelType
from .groq_model import GroqModel
from .gemini_model import GeminiModel


class TaskType(Enum):
    """Types of tasks for model routing."""
    FAST_EXTRACTION = "fast_extraction"  # Use faster model (Groq 8b)
    COMPLEX_REASONING = "complex_reasoning"  # Use best reasoning model (Gemini 2.5)
    STRUCTURED_OUTPUT = "structured_output"  # Either, prefer faster


class ModelManager:
    """
    Manages multiple AI models with routing and fallback.

    Routing Strategy (Groq-primary):
    - Fast extraction tasks -> Groq llama-3.1-8b-instant (fallback: Groq 70b)
    - Complex reasoning -> Groq 70b (fallback: Gemini if available)
    - Structured output -> Groq 70b (fallback: Gemini if available)
    - Automatic retry with exponential backoff on rate limits
    """

    def __init__(
        self,
        groq_api_key: str,
        google_api_key: str = "",
        groq_model: str = "llama-3.3-70b-versatile",
        groq_fast_model: str = "llama-3.1-8b-instant",
        gemini_model: str = "gemini-2.0-flash",
    ):
        # Groq models (primary)
        self.groq = GroqModel(groq_api_key, groq_model)
        self.groq_fast = GroqModel(groq_api_key, groq_fast_model)

        # Gemini as optional fallback (activate if key provided)
        if google_api_key and google_api_key not in ("", "your_google_api_key_here"):
            try:
                self.gemini = GeminiModel(google_api_key, gemini_model)
                self._gemini_available = True
            except Exception:
                self.gemini = self.groq
                self._gemini_available = False
        else:
            self.gemini = self.groq
            self._gemini_available = False

        # Track rate limiting
        self._groq_rate_limited = False
        self._rate_limit_reset_time = 0

    def _get_model_for_task(self, task_type: TaskType) -> tuple[BaseModel, BaseModel]:
        """Get primary and fallback model for task type.

        Groq is always primary. Gemini is used as fallback when available.
        """
        if task_type == TaskType.COMPLEX_REASONING:
            # Groq 70b primary, Gemini fallback (or Groq fast if no Gemini)
            if self._gemini_available:
                return self.groq, self.gemini
            return self.groq, self.groq_fast
        elif task_type == TaskType.STRUCTURED_OUTPUT:
            if self._gemini_available:
                return self.groq, self.gemini
            return self.groq, self.groq_fast
        else:  # FAST_EXTRACTION
            return self.groq_fast, self.groq

    async def _retry_with_backoff(
        self,
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ) -> ModelResponse:
        """Execute a model call with exponential backoff retry on rate limits."""
        last_response = None
        for attempt in range(max_retries):
            response = await func(**kwargs)
            if response.success:
                return response
            last_response = response

            # Only retry on rate limit errors
            if response.error and "rate" in response.error.lower():
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                break  # Non-rate-limit error, don't retry

        return last_response

    async def generate(
        self,
        prompt: str,
        task_type: TaskType = TaskType.FAST_EXTRACTION,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate response with automatic model selection, retry, and fallback."""
        primary, fallback = self._get_model_for_task(task_type)

        # Try primary model with retries
        response = await self._retry_with_backoff(
            primary.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response.success:
            return response

        # Track rate limiting
        if response.error and "rate" in response.error.lower():
            self._groq_rate_limited = True

        # Try fallback model with retries
        return await self._retry_with_backoff(
            fallback.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        task_type: TaskType = TaskType.STRUCTURED_OUTPUT,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Generate structured JSON with retry and fallback."""
        primary, fallback = self._get_model_for_task(task_type)

        # Try primary with retries
        response = await self._retry_with_backoff(
            primary.generate_structured,
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
        )

        if response.success:
            return response

        # Try fallback with retries
        return await self._retry_with_backoff(
            fallback.generate_structured,
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
