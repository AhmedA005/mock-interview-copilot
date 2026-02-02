"""
Configuration for the Streamlit frontend.
"""

import os


class Config:
    """Frontend configuration settings."""

    # API Configuration
    # Update these with your actual backend URL and API key
    API_URL: str = os.getenv(
        "API_URL",
        "http://localhost:8000/interview"
    )
    API_KEY: str = os.getenv("API_KEY", "secret123")

    # UI Configuration
    PAGE_TITLE: str = "Mock Interview Copilot"
    PAGE_ICON: str = "🤖"
    LAYOUT: str = "wide"

    # Request Configuration
    REQUEST_TIMEOUT: int = 300  # 5 minutes for model inference


config = Config()
