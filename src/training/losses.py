"""
Loss functions for neural network training
"""

import numpy as np
from typing import Callable


class LossFunctions:
    """
    Collection of loss functions and their gradients
    """

    @staticmethod
    def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Squared Error"""
        return np.mean((predictions - targets) ** 2)

    @staticmethod
    def mse_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """MSE gradient"""
        return 2 * (predictions - targets) / predictions.shape[0]

    @staticmethod
    def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(predictions - targets))

    @staticmethod
    def mae_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """MAE gradient"""
        return np.sign(predictions - targets) / predictions.shape[0]

    @staticmethod
    def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Binary Cross Entropy"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        return -np.mean(
            targets * np.log(predictions) +
            (1 - targets) * np.log(1 - predictions)
        )

    @staticmethod
    def binary_cross_entropy_grad(predictions: np.ndarray,
                                  targets: np.ndarray) -> np.ndarray:
        """Binary cross entropy gradient"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        return (predictions - targets) / (predictions * (1 - predictions)) / predictions.shape[0]

    @staticmethod
    def cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Categorical Cross Entropy"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        return -np.mean(np.sum(targets * np.log(predictions), axis=-1))

    @staticmethod
    def cross_entropy_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Cross entropy gradient"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        return (predictions - targets) / predictions.shape[0]

    @staticmethod
    def huber_loss(predictions: np.ndarray, targets: np.ndarray,
                   delta: float = 1.0) -> float:
        """Huber loss (smooth L1 loss)"""
        error = predictions - targets
        abs_error = np.abs(error)

        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic

        return np.mean(0.5 * quadratic**2 + delta * linear)

    @staticmethod
    def huber_loss_grad(predictions: np.ndarray, targets: np.ndarray,
                       delta: float = 1.0) -> np.ndarray:
        """Huber loss gradient"""
        error = predictions - targets
        abs_error = np.abs(error)

        grad = np.where(
            abs_error <= delta,
            error,
            delta * np.sign(error)
        )

        return grad / predictions.shape[0]

    @staticmethod
    def hinge_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Hinge loss for SVM-style classification"""
        return np.mean(np.maximum(0, 1 - targets * predictions))

    @staticmethod
    def hinge_loss_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Hinge loss gradient"""
        margin = 1 - targets * predictions
        grad = np.where(margin > 0, -targets, 0)
        return grad / predictions.shape[0]

    @staticmethod
    def kl_divergence(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Kullback-Leibler divergence"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        targets = np.clip(targets, 1e-7, 1 - 1e-7)
        return np.mean(np.sum(targets * np.log(targets / predictions), axis=-1))

    @staticmethod
    def kl_divergence_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """KL divergence gradient"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        targets = np.clip(targets, 1e-7, 1 - 1e-7)
        return -targets / predictions / predictions.shape[0]

    @staticmethod
    def focal_loss(predictions: np.ndarray, targets: np.ndarray,
                   alpha: float = 0.25, gamma: float = 2.0) -> float:
        """Focal loss for handling class imbalance"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        ce = -targets * np.log(predictions)
        weight = alpha * (1 - predictions) ** gamma
        return np.mean(weight * ce)

    @staticmethod
    def focal_loss_grad(predictions: np.ndarray, targets: np.ndarray,
                       alpha: float = 0.25, gamma: float = 2.0) -> np.ndarray:
        """Focal loss gradient"""
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        p = predictions
        y = targets

        grad = alpha * (
            y * (gamma * (1 - p) ** (gamma - 1) * np.log(p) + (1 - p) ** gamma / p) -
            (1 - y) * (gamma * p ** (gamma - 1) * np.log(1 - p) + p ** gamma / (1 - p))
        )

        return grad / predictions.shape[0]

    @staticmethod
    def cosine_similarity_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Cosine similarity loss"""
        predictions_norm = predictions / (np.linalg.norm(predictions, axis=-1, keepdims=True) + 1e-8)
        targets_norm = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)

        similarity = np.sum(predictions_norm * targets_norm, axis=-1)
        return np.mean(1 - similarity)

    @staticmethod
    def cosine_similarity_loss_grad(predictions: np.ndarray,
                                    targets: np.ndarray) -> np.ndarray:
        """Cosine similarity loss gradient"""
        pred_norm = np.linalg.norm(predictions, axis=-1, keepdims=True) + 1e-8
        target_norm = np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8

        predictions_normalized = predictions / pred_norm
        targets_normalized = targets / target_norm

        dot_product = np.sum(predictions_normalized * targets_normalized,
                            axis=-1, keepdims=True)

        grad = -(targets_normalized - dot_product * predictions_normalized) / pred_norm
        return grad / predictions.shape[0]

    @staticmethod
    def contrastive_loss(predictions: np.ndarray, targets: np.ndarray,
                        margin: float = 1.0) -> float:
        """Contrastive loss for metric learning"""
        euclidean_distance = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))

        loss = targets * euclidean_distance ** 2 + \
               (1 - targets) * np.maximum(0, margin - euclidean_distance) ** 2

        return np.mean(loss)

    @staticmethod
    def contrastive_loss_grad(predictions: np.ndarray, targets: np.ndarray,
                             margin: float = 1.0) -> np.ndarray:
        """Contrastive loss gradient"""
        diff = predictions - targets
        euclidean_distance = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True)) + 1e-8

        similar_grad = 2 * diff
        dissimilar_grad = -2 * np.maximum(0, margin - euclidean_distance) * diff / euclidean_distance

        grad = targets[..., None] * similar_grad + (1 - targets[..., None]) * dissimilar_grad

        return grad / predictions.shape[0]

    @staticmethod
    def get_loss(name: str) -> Callable:
        """Get loss function by name"""
        losses = {
            'mse': LossFunctions.mse,
            'mae': LossFunctions.mae,
            'binary_cross_entropy': LossFunctions.binary_cross_entropy,
            'cross_entropy': LossFunctions.cross_entropy,
            'huber': LossFunctions.huber_loss,
            'hinge': LossFunctions.hinge_loss,
            'kl_divergence': LossFunctions.kl_divergence,
            'focal': LossFunctions.focal_loss,
            'cosine': LossFunctions.cosine_similarity_loss,
            'contrastive': LossFunctions.contrastive_loss
        }
        return losses.get(name, LossFunctions.mse)

    @staticmethod
    def get_loss_grad(name: str) -> Callable:
        """Get loss gradient by name"""
        gradients = {
            'mse': LossFunctions.mse_grad,
            'mae': LossFunctions.mae_grad,
            'binary_cross_entropy': LossFunctions.binary_cross_entropy_grad,
            'cross_entropy': LossFunctions.cross_entropy_grad,
            'huber': LossFunctions.huber_loss_grad,
            'hinge': LossFunctions.hinge_loss_grad,
            'kl_divergence': LossFunctions.kl_divergence_grad,
            'focal': LossFunctions.focal_loss_grad,
            'cosine': LossFunctions.cosine_similarity_loss_grad,
            'contrastive': LossFunctions.contrastive_loss_grad
        }
        return gradients.get(name, LossFunctions.mse_grad)
