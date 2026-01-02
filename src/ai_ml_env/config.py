from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Global configuration for experiments."""

    project_name: str = "ai-ml-environment"
    data_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")
    seed: int = 42

    class Config:
        env_prefix = "AI_ML_"
        env_file = ".env"


def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
