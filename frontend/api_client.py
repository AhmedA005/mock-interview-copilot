"""
API client for communicating with the backend service.
"""

from typing import Dict, Optional, Tuple
import requests

from config import config


class APIClient:
    """Client for the Mock Interview Copilot API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_url = api_url or config.API_URL
        self.api_key = api_key or config.API_KEY
        self.timeout = config.REQUEST_TIMEOUT

    @property
    def base_url(self) -> str:
        """Get the base URL (without endpoint path)."""
        base = self.api_url
        for endpoint in ["/interview", "/generate-questions"]:
            if base.endswith(endpoint):
                base = base[: -len(endpoint)]
                break
        return base

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "MockInterviewCopilot/1.0",
        }

    def check_health(self) -> Tuple[bool, str]:
        """
        Check if the backend is healthy.
        
        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            response = requests.get(
                f"{self.base_url}/",
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=15,
            )
            if response.status_code == 200:
                return True, "Backend is online and ready"
            return False, f"Backend returned status {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except requests.exceptions.ConnectionError as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def generate_questions(
        self,
        resume_file,
        job_description: str,
    ) -> Tuple[bool, Dict | str]:
        """
        Generate interview questions.
        
        Args:
            resume_file: Uploaded PDF file object.
            job_description: Job description text.
            
        Returns:
            Tuple of (success, result/error_message)
        """
        files = {
            "file": (
                resume_file.name,
                resume_file.getvalue(),
                resume_file.type,
            )
        }
        data = {"job_description": job_description}

        try:
            response = requests.post(
                self.api_url,
                files=files,
                data=data,
                headers=self._get_headers(),
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return True, response.json()
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"Error ({response.status_code}): {response.text}"

        except requests.exceptions.Timeout:
            return False, "Request timed out. The model may be taking too long."
        except requests.exceptions.ConnectionError:
            return False, "Connection lost. Make sure the backend is running."
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
