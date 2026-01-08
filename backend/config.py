"""
Configuration management for Smart Study Buddy (Local Development Only)
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings for local development"""

    # API Configuration
    api_title: str = "Smart Study Buddy API"
    api_version: str = "1.0.0"
    api_description: str = "AI-powered study companion API"

    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

    # Security (simplified for local)
    secret_key: str = "local-development-key"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1000

    # Database Configuration (SQLite only)
    database_url: str = "sqlite+aiosqlite:///./studdy.db"

    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: str = ".pdf"
    upload_dir: str = "uploads"

    # ChromaDB Configuration
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "study_notes"

    # Logging Configuration
    log_level: str = "INFO"

    # Environment Configuration
    is_production: bool = False

    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_key(cls, v):
        if v and not (v.startswith('sk-') or v.startswith('sk-proj-')):
            raise ValueError('Invalid OpenAI API key format')
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_prefix": "",
        "extra": "ignore"
    }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

def validate_required_settings():
    """Validate that all required settings are present"""
    settings = get_settings()
    errors = []

    # For local development, make OpenAI API key optional
    # The app will work with fallback mechanisms
    if not settings.openai_api_key:
        print("Warning: OPENAI_API_KEY not set. App will use fallback mechanisms for AI features.")

    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

    return True
