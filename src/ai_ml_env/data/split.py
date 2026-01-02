from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Container for train/validation/test arrays."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def stratified_split(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> DataSplit:
    """Return a stratified train/val/test split."""

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        targets,
        test_size=val_size + test_size,
        stratify=targets,
        random_state=random_state,
    )

    relative_val_size = val_size / (val_size + test_size)

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1 - relative_val_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return DataSplit(x_train, x_val, x_test, y_train, y_val, y_test)
