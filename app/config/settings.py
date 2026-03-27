from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default=PROJECT_ROOT)
    gemini_api_key: str = Field(..., min_length=1)
    generation_model: str = "models/gemini-flash-lite-latest"
    embedding_model: str = "models/gemini-embedding-001"
    chroma_db_path: Path = Path("./chroma_db")
    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw")
    cache_db_path: Path = Path("./cache.db")
    ingestion_translation_sleep_seconds: float = 0.05
    embedding_batch_size: int = 100
    embedding_sleep_seconds: float = 0.2
    rag_generation_sleep_seconds: float = 4.2
    question_generation_sleep_seconds: float = 4.2

    @field_validator("chroma_db_path", "data_dir", "raw_data_dir", "cache_db_path", mode="before")
    @classmethod
    def _resolve_paths(cls, value: Path | str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (PROJECT_ROOT / candidate).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def ensure_runtime_dirs(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
    settings.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
