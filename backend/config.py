from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralised configuration for backend + Kaggle notebook."""

    model_id: str
    api_key: str
    ngrok_token: str | None
    device_map: str
    quant_in_4bit: bool

    def __init__(self) -> None:
        self.model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.api_key = os.getenv("API_KEY", "secret123")
        self.ngrok_token = os.getenv("NGROK_TOKEN")
        self.device_map = os.getenv("DEVICE_MAP", "auto")
        self.quant_in_4bit = os.getenv("QUANT_4BIT", "1") not in {"0", "false", "False"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

