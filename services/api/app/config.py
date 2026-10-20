from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "ml" / "artifacts" / "model.joblib"


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or services/api/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "production", "test"] = "development"
    model_path: Path = DEFAULT_MODEL_PATH
    cors_origins: str = "http://localhost:3000"
    enable_demo_mode: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def metadata_path(self) -> Path:
        """Sidecar JSON written next to the model artifact by the training script."""
        return self.model_path.with_name(f"{self.model_path.stem}.metadata.json")


@lru_cache
def get_settings() -> Settings:
    return Settings()
