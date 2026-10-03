"""Application settings, loaded from environment variables / a local ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- storage ---
    DATABASE_URL: str = "sqlite:///./data/lca_store.db"
    LIFE_CYCLE_TSV_PATH: str = "data/life_cycle.tsv"
    ARTIFACTS_DIR: str = "artifacts"

    # --- external upstream service ---
    EXTERNAL_DATA_API_URL: str | None = None
    EXTERNAL_DATA_API_KEY: str | None = None
    EXTERNAL_DATA_API_TIMEOUT: float = 30.0

    # --- continual learning ---
    RETRAIN_THRESHOLD_NEW_RECORDS: int = 50
    RETRAIN_CRON_SCHEDULE: str = "0 2 * * *"  # 2 AM daily; empty string disables the cron job
    SYNC_POLL_INTERVAL_MINUTES: int = 15  # threshold check / external poll; 0 disables
    ENABLE_SCHEDULER: bool = True

    # --- training guards ---
    MIN_TRAIN_SAMPLES: int = 50
    MIN_AVG_R2: float = 0.0  # validation gate: average test R^2 must be >= this
    MAX_R2_REGRESSION: float = 0.05  # candidate may not trail the active model by more than this

    # --- API ---
    CORS_ALLOW_ORIGINS: str = ""  # comma-separated browser origins, e.g. http://localhost:8080
    ADMIN_API_KEY: str | None = None  # when set, mutating endpoints require X-API-Key

    @property
    def artifacts_path(self) -> Path:
        return Path(self.ARTIFACTS_DIR)

    @property
    def models_path(self) -> Path:
        return self.artifacts_path / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
