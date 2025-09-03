"""
Application settings and configuration management.

This module provides centralized configuration management using Pydantic Settings.
It handles environment variables, default values, and validation for all
application settings including model configurations, API settings, and paths.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration.

    This class manages all application settings using Pydantic Settings.
    It provides type validation, environment variable loading, and
    default values for all configuration parameters.
    """

    # Embedding model configuration
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the sentence transformer model for embeddings",
    )

    # Vector index (FAISS) configuration
    top_k: int = Field(
        default=3, description="Number of top similar documents to retrieve"
    )
    rebuild_index: bool = Field(
        default=False, description="Whether to rebuild the FAISS index from scratch"
    )

    # LLM configuration
    llm_provider: str = Field(
        default="openai", description="LLM provider (currently supports 'openai')"
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name")
    llm_temperature: float = Field(
        default=0.0, description="LLM temperature for response generation"
    )
    llm_max_tokens: int = Field(
        default=512, description="Maximum tokens for LLM response"
    )
    llm_top_p: float = Field(
        default=1.0, description="LLM top_p parameter for nucleus sampling"
    )
    llm_frequency_penalty: float = Field(
        default=0.0, description="LLM frequency penalty to reduce repetition"
    )
    llm_presence_penalty: float = Field(
        default=0.0, description="LLM presence penalty to encourage new topics"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }

    # LLM Response Constraints
    allowed_actions: list[str] = Field(
        default=[
            "none",
            "escalate_to_support",
            "escalate_to_abuse_team",
            "contact_customer",
        ],
        description="Allowed action types for LLM responses",
    )
    max_references: int = Field(
        default=3, description="Maximum number of document references in responses"
    )

    # API configuration
    api_host: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API host URL"
    )
    api_key: Optional[str] = Field(
        default=None, description="OpenAI API key (loaded from environment)"
    )
    api_port: int = Field(default=8000, description="Port for the FastAPI server")

    # Security tokens
    reindex_token: str = Field(
        default="dev-reindex-2025",
        description="Token for reindex endpoint authentication",
    )
    debug_token: str = Field(
        default="dev-debug-2025", description="Token for debug endpoints authentication"
    )

    # File paths
    docs_path: str = Field(
        default="src/data/docs.json",
        description="Path to the knowledge base documents file",
    )
    index_path: str = Field(
        default="index/faiss_index.bin", description="Path to the FAISS index file"
    )

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        """Validate top_k is a positive integer."""
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("llm_temperature must be between 0 and 2")
        return v

    @field_validator("llm_max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate max_tokens is a positive integer."""
        if v <= 0:
            raise ValueError("llm_max_tokens must be a positive integer")
        return v

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, v):
        """Validate API port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("api_port must be between 1 and 65535")
        return v


# Load environment variables from .env file
load_dotenv()

# Create settings instance after loading environment variables
settings = Settings()

# Manually set api_key from environment variable if Pydantic didn't load it
if not settings.api_key and os.getenv("OPENAI_API_KEY"):
    settings.api_key = os.getenv("OPENAI_API_KEY")

# Validate critical environment variables
if not settings.api_key:
    print("Warning: OPENAI_API_KEY environment variable is not set")
    print("Please create a .env file with your OpenAI API key:")
    print("OPENAI_API_KEY=your_actual_api_key_here")
else:
    print("OpenAI API key loaded successfully")
