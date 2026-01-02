from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .architectures import build_classification_pipeline


class Trainer:
    def __init__(self, model_name: str = 'logistic_regression', **model_kwargs: Any) -> None:
        self.pipeline = build_classification_pipeline(model_name=model_name, **model_kwargs)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.pipeline.fit(features, target)

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        predictions = self.pipeline.predict(features)
        return {
            'accuracy': accuracy_score(target, predictions),
            'report': classification_report(target, predictions, output_dict=True),
        }

    def predict(self, features: pd.DataFrame) -> Any:
        return self.pipeline.predict(features)

    def save(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, output_path)
        return output_path

    @classmethod
    def load(cls, model_path: str | Path) -> 'Trainer':
        pipeline = joblib.load(model_path)
        instance = cls.__new__(cls)
        instance.pipeline = pipeline
        return instance
