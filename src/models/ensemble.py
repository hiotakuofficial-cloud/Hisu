"""
Ensemble learning models combining multiple ML models.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator


class DeepEnsemble:
    """Ensemble of neural network models."""

    def __init__(self, models: List[nn.Module]):
        """
        Initialize deep ensemble.

        Args:
            models: List of PyTorch models
        """
        self.models = models
        self.num_models = len(models)

    def predict(self, x: torch.Tensor, return_std: bool = False) -> Union[torch.Tensor, tuple]:
        """
        Make predictions using ensemble.

        Args:
            x: Input tensor
            return_std: Whether to return standard deviation

        Returns:
            Mean predictions (and std if return_std=True)
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)

        if return_std:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred

        return mean_pred

    def train_model(self, model_idx: int):
        """Set specific model to training mode."""
        self.models[model_idx].train()

    def eval_models(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            model.eval()


class SklearnEnsemble:
    """Ensemble using scikit-learn models."""

    def __init__(
        self,
        model_type: str = 'classifier',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize sklearn ensemble.

        Args:
            model_type: 'classifier' or 'regressor'
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.model_type = model_type
        self.models = []

        if model_type == 'classifier':
            self.models.append(
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            )
            self.models.append(
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else 3,
                    random_state=random_state
                )
            )
        else:
            self.models.append(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            )
            self.models.append(
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else 3,
                    random_state=random_state
                )
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train all models in ensemble.

        Args:
            X: Training features
            y: Training labels
        """
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble averaging.

        Args:
            X: Input features

        Returns:
            Averaged predictions
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)
        return predictions.mean(axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classifiers).

        Args:
            X: Input features

        Returns:
            Averaged probability predictions
        """
        if self.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifiers")

        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)

        predictions = np.array(predictions)
        return predictions.mean(axis=0)


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""

    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: BaseEstimator
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base models
            meta_model: Meta-learner model
        """
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train stacking ensemble.

        Args:
            X: Training features
            y: Training labels
        """
        # Train base models
        base_predictions = []
        for model in self.base_models:
            model.fit(X, y)
            pred = model.predict(X)
            base_predictions.append(pred)

        # Stack predictions for meta-learner
        meta_features = np.column_stack(base_predictions)

        # Train meta-learner
        self.meta_model.fit(meta_features, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.

        Args:
            X: Input features

        Returns:
            Final predictions from meta-learner
        """
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)

        # Stack predictions
        meta_features = np.column_stack(base_predictions)

        # Meta-learner prediction
        return self.meta_model.predict(meta_features)
