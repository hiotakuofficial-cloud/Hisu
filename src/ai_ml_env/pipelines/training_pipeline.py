from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from ..config.settings import MODELS_DIR
from ..data.datamodule import load_dataset, split_features_targets, train_validation_split
from ..models.trainer import Trainer


def run_training(csv_path: str, target_column: str, model_name: str = 'logistic_regression', **model_kwargs: Any) -> Dict[str, Any]:
    dataset = load_dataset(csv_path)
    features, target = split_features_targets(dataset, target_column)
    features_train, features_val, target_train, target_val = train_validation_split(features, target)

    trainer = Trainer(model_name=model_name, **model_kwargs)
    trainer.fit(features_train, target_train)
    metrics = trainer.evaluate(features_val, target_val)

    model_path = MODELS_DIR / f"{model_name}.joblib"
    trainer.save(model_path)

    return {
        'metrics': metrics,
        'model_path': model_path,
    }
