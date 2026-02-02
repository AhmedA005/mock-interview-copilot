"""
LLM (Large Language Model) management module.
Handles loading and inference with the Qwen model.
"""

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import settings


class LLMManager:
    """Manages the LLM model lifecycle and inference."""

    _instance: Optional["LLMManager"] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModelForCausalLM] = None

    def __new__(cls) -> "LLMManager":
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._tokenizer is not None and self._model is not None

    def load(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load or return the cached model and tokenizer."""
        if self.is_loaded:
            return self._tokenizer, self._model

        print(f"🚀 Loading LLM: {settings.MODEL_ID}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_ID, use_fast=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        ).eval()

        print("✅ LLM loaded successfully")
        return self._tokenizer, self._model

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text using the LLM."""
        if not self.is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
        temperature = temperature or settings.TEMPERATURE
        top_p = top_p or settings.TOP_P

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response if present
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]

        return response.strip()


# Global instance for easy access
llm_manager = LLMManager()
