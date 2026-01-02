from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import get_settings


def load_csv(name: str, *, transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> pd.DataFrame:
    """Load a CSV dataset from the configured data directory."""

    settings = get_settings()
    data_path = Path(settings.data_dir) / f"{name}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if transform is not None:
        df = transform(df)

    return df


def save_dataframe(df: pd.DataFrame, name: str) -> Path:
    """Persist a dataframe within the processed data directory."""

    settings = get_settings()
    processed_dir = Path(settings.data_dir).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / f"{name}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path
