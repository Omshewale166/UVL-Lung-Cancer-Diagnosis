"""Typed settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration; paths are relative to the working directory."""

    model_config = SettingsConfigDict(env_prefix="LUNG_", env_file=".env", extra="ignore")

    model_path: Path = Path("artifacts/model.weights.h5")
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    random_seed: int = 42
    api_key: str | None = None

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Validate a standard Python logging level."""
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log_level must be a standard Python logging level")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
