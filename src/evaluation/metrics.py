"""
Evaluation metrics for neural network models
"""

import numpy as np
from typing import Dict, Any, Optional, List


class Metrics:
    """
    Base metrics class
    """

    @staticmethod
    def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute accuracy

        Args:
            predictions: Predicted labels
            targets: True labels

        Returns:
            Accuracy score
        """
        return np.mean(predictions == targets)

    @staticmethod
    def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute MSE

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            MSE score
        """
        return np.mean((predictions - targets) ** 2)

    @staticmethod
    def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute MAE

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            MAE score
        """
        return np.mean(np.abs(predictions - targets))

    @staticmethod
    def root_mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute RMSE"""
        return np.sqrt(Metrics.mean_squared_error(predictions, targets))

    @staticmethod
    def r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute R² score

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            R² score
        """
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))


class ClassificationMetrics:
    """
    Metrics for classification tasks
    """

    @staticmethod
    def confusion_matrix(predictions: np.ndarray, targets: np.ndarray,
                        num_classes: Optional[int] = None) -> np.ndarray:
        """
        Compute confusion matrix

        Args:
            predictions: Predicted class labels
            targets: True class labels
            num_classes: Number of classes

        Returns:
            Confusion matrix
        """
        if num_classes is None:
            num_classes = max(np.max(predictions), np.max(targets)) + 1

        cm = np.zeros((num_classes, num_classes), dtype=np.int32)

        for pred, target in zip(predictions.flatten(), targets.flatten()):
            cm[int(target), int(pred)] += 1

        return cm

    @staticmethod
    def precision(predictions: np.ndarray, targets: np.ndarray,
                 average: str = 'macro') -> float:
        """
        Compute precision

        Args:
            predictions: Predicted labels
            targets: True labels
            average: 'macro', 'micro', or 'weighted'

        Returns:
            Precision score
        """
        cm = ClassificationMetrics.confusion_matrix(predictions, targets)
        num_classes = cm.shape[0]

        if average == 'micro':
            tp = np.sum(np.diag(cm))
            fp = np.sum(cm) - tp
            return tp / (tp + fp + 1e-8)

        precisions = []
        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            precisions.append(tp / (tp + fp + 1e-8))

        if average == 'macro':
            return np.mean(precisions)
        elif average == 'weighted':
            weights = np.sum(cm, axis=1)
            return np.average(precisions, weights=weights)

        return np.array(precisions)

    @staticmethod
    def recall(predictions: np.ndarray, targets: np.ndarray,
              average: str = 'macro') -> float:
        """
        Compute recall

        Args:
            predictions: Predicted labels
            targets: True labels
            average: 'macro', 'micro', or 'weighted'

        Returns:
            Recall score
        """
        cm = ClassificationMetrics.confusion_matrix(predictions, targets)
        num_classes = cm.shape[0]

        if average == 'micro':
            tp = np.sum(np.diag(cm))
            fn = np.sum(cm) - tp
            return tp / (tp + fn + 1e-8)

        recalls = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            recalls.append(tp / (tp + fn + 1e-8))

        if average == 'macro':
            return np.mean(recalls)
        elif average == 'weighted':
            weights = np.sum(cm, axis=1)
            return np.average(recalls, weights=weights)

        return np.array(recalls)

    @staticmethod
    def f1_score(predictions: np.ndarray, targets: np.ndarray,
                average: str = 'macro') -> float:
        """
        Compute F1 score

        Args:
            predictions: Predicted labels
            targets: True labels
            average: 'macro', 'micro', or 'weighted'

        Returns:
            F1 score
        """
        precision = ClassificationMetrics.precision(predictions, targets, average)
        recall = ClassificationMetrics.recall(predictions, targets, average)

        return 2 * (precision * recall) / (precision + recall + 1e-8)

    @staticmethod
    def auc_roc(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute AUC-ROC score

        Args:
            predictions: Predicted probabilities
            targets: True binary labels

        Returns:
            AUC-ROC score
        """
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_targets = targets[sorted_indices]

        tpr = np.cumsum(sorted_targets) / np.sum(sorted_targets)
        fpr = np.cumsum(1 - sorted_targets) / np.sum(1 - sorted_targets)

        auc = np.trapz(tpr, fpr)
        return abs(auc)

    @staticmethod
    def log_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute log loss

        Args:
            predictions: Predicted probabilities
            targets: True labels

        Returns:
            Log loss
        """
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        return -np.mean(
            targets * np.log(predictions) +
            (1 - targets) * np.log(1 - predictions)
        )

    @staticmethod
    def top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
        """
        Compute top-k accuracy

        Args:
            predictions: Predicted probabilities (batch_size, num_classes)
            targets: True class labels
            k: Top k predictions to consider

        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(predictions, axis=-1)[:, -k:]
        targets = targets.reshape(-1, 1)

        correct = np.any(top_k_preds == targets, axis=-1)
        return np.mean(correct)

    @staticmethod
    def balanced_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute balanced accuracy

        Args:
            predictions: Predicted labels
            targets: True labels

        Returns:
            Balanced accuracy
        """
        cm = ClassificationMetrics.confusion_matrix(predictions, targets)
        num_classes = cm.shape[0]

        recalls = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            recalls.append(tp / (tp + fn + 1e-8))

        return np.mean(recalls)

    @staticmethod
    def matthews_corrcoef(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Matthews correlation coefficient

        Args:
            predictions: Predicted binary labels
            targets: True binary labels

        Returns:
            MCC score
        """
        tp = np.sum((predictions == 1) & (targets == 1))
        tn = np.sum((predictions == 0) & (targets == 0))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))

        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + 1e-8)


class RegressionMetrics:
    """
    Metrics for regression tasks
    """

    @staticmethod
    def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute explained variance

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            Explained variance score
        """
        var_y = np.var(targets)
        var_residual = np.var(targets - predictions)
        return 1 - (var_residual / (var_y + 1e-8))

    @staticmethod
    def max_error(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute maximum residual error

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            Maximum error
        """
        return np.max(np.abs(targets - predictions))

    @staticmethod
    def median_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute median absolute error

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            Median absolute error
        """
        return np.median(np.abs(targets - predictions))

    @staticmethod
    def mean_absolute_percentage_error(predictions: np.ndarray,
                                      targets: np.ndarray) -> float:
        """
        Compute MAPE

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            MAPE score
        """
        return np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    @staticmethod
    def symmetric_mape(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute symmetric MAPE

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            SMAPE score
        """
        numerator = np.abs(targets - predictions)
        denominator = (np.abs(targets) + np.abs(predictions)) / 2
        return np.mean(numerator / (denominator + 1e-8)) * 100

    @staticmethod
    def mean_squared_log_error(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute mean squared logarithmic error

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            MSLE score
        """
        log_pred = np.log1p(np.maximum(predictions, 0))
        log_target = np.log1p(np.maximum(targets, 0))
        return np.mean((log_pred - log_target) ** 2)

    @staticmethod
    def huber_metric(predictions: np.ndarray, targets: np.ndarray,
                    delta: float = 1.0) -> float:
        """
        Compute Huber metric

        Args:
            predictions: Predicted values
            targets: True values
            delta: Threshold

        Returns:
            Huber metric
        """
        error = np.abs(targets - predictions)
        quadratic = np.minimum(error, delta)
        linear = error - quadratic
        return np.mean(0.5 * quadratic**2 + delta * linear)
