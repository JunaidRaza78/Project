"""
Autonomous Research Agent Configuration

Centralized configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""

import os
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

    # LangSmith Configuration
    langchain_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key for monitoring",
    )
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langchain_project: str = Field(
        default="research-agent",
        description="LangSmith project name",
    )

    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: Optional[str] = Field(
        default=None,
        description="Neo4j password",
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

    def setup_langsmith(self) -> None:
        """Set LangSmith environment variables if configured."""
        if self.langchain_api_key and self.langchain_api_key != "your_langsmith_api_key_here":
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.langchain_tracing_v2).lower()
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
