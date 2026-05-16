"""
Main application entry point.
Creates and configures the FastAPI application.
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path

import nest_asyncio
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .config import settings
from .models import LLMManager

# Apply nest_asyncio for Jupyter/Kaggle compatibility
nest_asyncio.apply()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Mock Interview Copilot API",
        description="AI-powered interview question generation",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    return app


def start_ngrok(port: int):
    """Start ngrok tunnel for public access."""
    binary_path = Path(settings.NGROK_BINARY)
    
    if not binary_path.exists():
        resolved = shutil.which("ngrok")
        if not resolved:
            print("⚠️ ngrok binary not found; skipping tunnel setup.")
            return
        binary_path = Path(resolved)

    if settings.NGROK_AUTHTOKEN:
        subprocess.run(
            [str(binary_path), "config", "add-authtoken", settings.NGROK_AUTHTOKEN],
            check=False,
        )

    process = subprocess.Popen(
        [str(binary_path), "http", str(port), "--log=stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    time.sleep(4)
    
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        tunnels = response.json()
        public_url = tunnels["tunnels"][0]["public_url"]
        
        print("\n" + "=" * 70)
        print("🚀 MOCK INTERVIEW COPILOT API IS RUNNING!")
        print("=" * 70)
        print(f"📱 Public URL: {public_url}")
        print(f"🏠 Local URL:  http://localhost:{port}")
        print(f"🔑 API Key:    {settings.API_KEY}")
        print("=" * 70 + "\n")
    except Exception as exc:
        print(f"⚠️ Unable to query ngrok tunnel: {exc}")


def run():
    """Run the application."""
    # Pre-load the LLM
    llm = LLMManager()
    llm.load()

    app = create_app()
    port = settings.PORT

    # Start ngrok if enabled
    if settings.ENABLE_NGROK:
        thread = threading.Thread(target=start_ngrok, args=(port,), daemon=True)
        thread.start()
        time.sleep(5)

    print(f"\n🚀 Starting server on {settings.HOST}:{port}")
    uvicorn.run(app, host=settings.HOST, port=port, log_level="info")


# Create app instance for ASGI servers
app = create_app()


if __name__ == "__main__":
    run()
