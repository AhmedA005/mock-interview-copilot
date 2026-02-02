"""
Configuration settings for the Mock Interview Copilot API.
All environment variables and constants are centralized here.
"""

import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    # API Configuration
    API_KEY: str = os.getenv("API_KEY", "secret123")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Model Configuration
    MODEL_ID: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    EMBED_MODEL_NAME: str = os.getenv(
        "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Generation Settings
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "2100"))
    GENERATION_RETRIES: int = int(os.getenv("GENERATION_RETRIES", "2"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))

    # Resume Processing
    RESUME_MAX_LENGTH: int = int(os.getenv("RESUME_MAX_LENGTH", "1500"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "40"))

    # ngrok Configuration (for Kaggle deployment)
    ENABLE_NGROK: bool = os.getenv("ENABLE_NGROK", "0") not in {"0", "false", "False"}
    NGROK_BINARY: str = os.getenv("NGROK_BINARY", "./ngrok")
    NGROK_AUTHTOKEN: Optional[str] = os.getenv("NGROK_AUTHTOKEN")

    # Question Generation Defaults
    DEFAULT_TECHNICAL_QUESTIONS: int = 5
    DEFAULT_BEHAVIORAL_QUESTIONS: int = 3


settings = Settings()
