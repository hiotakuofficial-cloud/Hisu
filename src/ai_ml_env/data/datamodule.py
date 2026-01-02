from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config.settings import DEFAULT_RANDOM_STATE, TRAIN_TEST_SPLIT


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_features_targets(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def train_validation_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size or TRAIN_TEST_SPLIT,
        random_state=random_state or DEFAULT_RANDOM_STATE,
    )
