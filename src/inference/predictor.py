"""
Inference and prediction utilities for trained models.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, List
from pathlib import Path


class TorchPredictor:
    """Predictor for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize predictor.

        Args:
            model: PyTorch model
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input data
            batch_size: Batch size for inference

        Returns:
            Predictions as numpy array
        """
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size].to(self.device)
                output = self.model(batch)
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def predict_proba(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input data
            batch_size: Batch size for inference

        Returns:
            Probability predictions
        """
        predictions = self.predict(X, batch_size)
        return torch.softmax(torch.FloatTensor(predictions), dim=1).numpy()

    def predict_single(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict on a single sample.

        Args:
            x: Single input sample

        Returns:
            Prediction
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)

        return output.cpu().numpy()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> 'TorchPredictor':
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            model: Model architecture (uninitialized)
            device: Device to load model on

        Returns:
            Initialized predictor
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return cls(model, device)


class EnsemblePredictor:
    """Predictor for ensemble of models."""

    def __init__(self, predictors: List[TorchPredictor]):
        """
        Initialize ensemble predictor.

        Args:
            predictors: List of individual predictors
        """
        self.predictors = predictors

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
        return_std: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Make ensemble predictions.

        Args:
            X: Input data
            batch_size: Batch size for inference
            return_std: Whether to return standard deviation

        Returns:
            Mean predictions (and std if return_std=True)
        """
        all_predictions = []
        for predictor in self.predictors:
            pred = predictor.predict(X, batch_size)
            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)
        mean_pred = all_predictions.mean(axis=0)

        if return_std:
            std_pred = all_predictions.std(axis=0)
            return mean_pred, std_pred

        return mean_pred

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32
    ) -> tuple:
        """
        Predict with uncertainty estimation.

        Args:
            X: Input data
            batch_size: Batch size for inference

        Returns:
            (predictions, uncertainty) tuple
        """
        return self.predict(X, batch_size, return_std=True)


class BatchPredictor:
    """Efficient batch prediction with memory management."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_batch_size: int = 1024
    ):
        """
        Initialize batch predictor.

        Args:
            model: PyTorch model
            device: Device to run inference on
            max_batch_size: Maximum batch size
        """
        self.predictor = TorchPredictor(model, device)
        self.max_batch_size = max_batch_size

    def predict_large_dataset(
        self,
        X: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Predict on large dataset with progress tracking.

        Args:
            X: Input data
            show_progress: Whether to show progress

        Returns:
            Predictions
        """
        predictions = []
        total_batches = (len(X) + self.max_batch_size - 1) // self.max_batch_size

        for i in range(0, len(X), self.max_batch_size):
            batch = X[i:i + self.max_batch_size]
            pred = self.predictor.predict(batch, batch_size=min(32, len(batch)))
            predictions.append(pred)

            if show_progress:
                batch_num = i // self.max_batch_size + 1
                print(f"Processing batch {batch_num}/{total_batches}", end='\r')

        if show_progress:
            print("\nPrediction complete!")

        return np.concatenate(predictions, axis=0)
