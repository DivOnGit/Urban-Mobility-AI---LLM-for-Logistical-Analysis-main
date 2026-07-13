from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from config import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelLoader:
    model_name: str = settings.model_name

    def load(self) -> HuggingFacePipeline:
        LOGGER.info("Loading model %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": settings.device_map,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if settings.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            do_sample=settings.temperature > 0,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=generator)
