"""
Autonomous Research Agent Configuration

Centralized configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Keys
    groq_api_key: str = Field(..., description="Groq API key for LLM access")
    google_api_key: str = Field(..., description="Google Gemini API key")
    serper_api_key: str = Field(..., description="Serper API key for search")
    
    # Model Configuration
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use"
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model to use"
    )
    
    # Agent Configuration
    max_search_iterations: int = Field(
        default=10,
        description="Maximum search iterations per investigation"
    )
    min_confidence_threshold: float = Field(
        default=0.3,
        description="Minimum confidence score to include findings"
    )
    
    # Logging
    debug_mode: bool = Field(
        default=False,
        description="Enable verbose debug logging"
    )
    
    # Paths
    output_dir: Path = Field(
        default=Path("output"),
        description="Output directory for reports and logs"
    )
    
    @property
    def reports_dir(self) -> Path:
        """Path to reports directory."""
        return self.output_dir / "reports"
    
    @property
    def logs_dir(self) -> Path:
        """Path to logs directory."""
        return self.output_dir / "logs"
    
    def ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
