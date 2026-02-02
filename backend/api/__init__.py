"""
API package - FastAPI routes and middleware.
"""

from .routes import router
from .auth import verify_api_key

__all__ = ["router", "verify_api_key"]
