"""
Groq Model Integration

Fast inference using Groq's LPU for llama models.
Primary model for fact extraction and initial analysis.
"""

import time
import json
from typing import Any, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .base_model import BaseModel, ModelResponse, ModelType


class GroqModel(BaseModel):
    """Groq API integration for fast LLM inference."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        super().__init__(api_key, model_name, ModelType.GROQ)
        self.client = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate text response using Groq."""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # Update client settings
            self.client.temperature = temperature
            self.client.max_tokens = max_tokens
            
            response = await self.client.ainvoke(messages)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=response.content,
                model_type=self.model_type,
                model_name=self.model_name,
                latency_ms=latency_ms,
                tokens_used=response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
                raw_response=response,
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                model_type=self.model_type,
                model_name=self.model_name,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Generate structured JSON response."""
        # Enhance prompt with JSON instructions
        json_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Do not include any text outside the JSON object."""
        
        enhanced_system = system_prompt or ""
        enhanced_system += "\nYou are a precise data extraction assistant. Always respond with valid JSON only."
        
        response = await self.generate(
            prompt=json_prompt,
            system_prompt=enhanced_system,
            temperature=0.3,  # Lower temp for structured output
        )
        
        # Try to parse and validate JSON
        if response.success:
            try:
                # Extract JSON from response
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                
                json.loads(content.strip())  # Validate JSON
                response.content = content.strip()
            except json.JSONDecodeError as e:
                response.error = f"Invalid JSON response: {e}"
        
        return response
