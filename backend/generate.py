from __future__ import annotations

import logging
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import get_settings

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_tokenizer(model_id: str):
    LOGGER.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def _load_model(model_id: str, device_map: str, quant_4bit: bool):
    LOGGER.info("Loading model: %s (quant_4bit=%s)", model_id, quant_4bit)
    quant_cfg = None
    if quant_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        quantization_config=quant_cfg,
    ).eval()
    return model


def get_tokenizer_model():
    settings = get_settings()
    tokenizer = _load_tokenizer(settings.model_id)
    model = _load_model(settings.model_id, settings.device_map, settings.quant_in_4bit)
    return tokenizer, model


def generate_text(prompt: str, max_new_tokens: int = 1200) -> str:
    tokenizer, model = get_tokenizer_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

