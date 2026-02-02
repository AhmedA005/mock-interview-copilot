"""
Authentication utilities for the API.
"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings

security = HTTPBearer()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verify the API key from the Authorization header.
    
    Args:
        credentials: Bearer token credentials.
        
    Returns:
        The verified API key.
        
    Raises:
        HTTPException: If the API key is invalid.
    """
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
