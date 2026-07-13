from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    data_path: Path = Path(os.getenv("DATA_PATH", "data/nyc_taxi.csv"))
    faiss_index_path: Path = Path(os.getenv("FAISS_INDEX_PATH", "vector_db/faiss_index"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "768"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    use_4bit: bool = _bool_env("USE_4BIT", True)
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    project_name: str = "Urban Mobility AI - DeepSeek-R1-Distill-LLaMA-70B"


settings = Settings()

