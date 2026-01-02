"""
Model evaluation metrics for various ML tasks.
"""
import numpy as np
from typing import Dict, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, matthews_corrcoef
)


class ClassificationMetrics:
    """Metrics for classification tasks."""

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging strategy for multiclass

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }

        if y_proba is not None:
            try:
                if average == 'binary':
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
                    metrics['log_loss'] = log_loss(y_true, y_proba)
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                pass

        return metrics

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Get detailed classification report."""
        return classification_report(y_true, y_pred, target_names=target_names)

    @staticmethod
    def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Specificity score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate balanced accuracy.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Balanced accuracy score
        """
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred)


class RegressionMetrics:
    """Metrics for regression tasks."""

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred)
        }

        return metrics

    @staticmethod
    def mean_absolute_percentage_error(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAPE score
        """
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def root_mean_squared_log_error(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Root Mean Squared Logarithmic Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RMSLE score
        """
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

    @staticmethod
    def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate explained variance score.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Explained variance score
        """
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(y_true, y_pred)


class ClusteringMetrics:
    """Metrics for clustering tasks."""

    @staticmethod
    def calculate_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score.

        Args:
            X: Feature data
            labels: Cluster labels

        Returns:
            Silhouette score
        """
        from sklearn.metrics import silhouette_score
        return silhouette_score(X, labels)

    @staticmethod
    def calculate_davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin score.

        Args:
            X: Feature data
            labels: Cluster labels

        Returns:
            Davies-Bouldin score
        """
        from sklearn.metrics import davies_bouldin_score
        return davies_bouldin_score(X, labels)

    @staticmethod
    def calculate_calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Calinski-Harabasz score.

        Args:
            X: Feature data
            labels: Cluster labels

        Returns:
            Calinski-Harabasz score
        """
        from sklearn.metrics import calinski_harabasz_score
        return calinski_harabasz_score(X, labels)


class MetricsTracker:
    """Track metrics across training epochs."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = {}

    def update(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Update metrics for an epoch.

        Args:
            metrics: Dictionary of metric values
            epoch: Epoch number
        """
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append((epoch, value))

    def get_history(self, metric_name: str) -> List[tuple]:
        """Get history for a specific metric."""
        return self.metrics_history.get(metric_name, [])

    def get_best_epoch(self, metric_name: str, mode: str = 'max') -> int:
        """
        Get epoch with best metric value.

        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'

        Returns:
            Best epoch number
        """
        history = self.metrics_history.get(metric_name, [])
        if not history:
            return 0

        if mode == 'max':
            best_epoch, _ = max(history, key=lambda x: x[1])
        else:
            best_epoch, _ = min(history, key=lambda x: x[1])

        return best_epoch

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary with min, max, mean for each metric
        """
        summary = {}
        for metric_name, history in self.metrics_history.items():
            values = [val for _, val in history]
            summary[metric_name] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        return summary
