"""
Google Gemini Model Integration

Used for complex reasoning and fallback when Groq is rate-limited.
"""

import time
import json
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base_model import BaseModel, ModelResponse, ModelType


class GeminiModel(BaseModel):
    """Google Gemini API integration."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-pro",
    ):
        super().__init__(api_key, model_name, ModelType.GEMINI)
        self.client = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate text response using Gemini."""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # Update client settings
            self.client.temperature = temperature
            self.client.max_output_tokens = max_tokens
            
            response = await self.client.ainvoke(messages)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract token usage if available
            tokens_used = 0
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("usage_metadata", {})
                tokens_used = usage.get("total_token_count", 0)
            
            return ModelResponse(
                content=response.content,
                model_type=self.model_type,
                model_name=self.model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
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
        json_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Do not include any text outside the JSON object. No markdown formatting."""
        
        enhanced_system = system_prompt or ""
        enhanced_system += "\nYou are a precise data extraction assistant. Always respond with valid JSON only, no markdown code blocks."
        
        response = await self.generate(
            prompt=json_prompt,
            system_prompt=enhanced_system,
            temperature=0.3,
        )
        
        if response.success:
            try:
                content = response.content.strip()
                # Clean up any markdown formatting
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                
                json.loads(content.strip())
                response.content = content.strip()
            except json.JSONDecodeError as e:
                response.error = f"Invalid JSON response: {e}"
        
        return response
