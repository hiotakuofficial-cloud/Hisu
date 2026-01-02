from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classification_pipeline(model_name: str = 'logistic_regression', **kwargs: Any) -> Pipeline:
    if model_name == 'logistic_regression':
        classifier = LogisticRegression(**kwargs)
    elif model_name == 'random_forest':
        classifier = RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', classifier),
    ])
