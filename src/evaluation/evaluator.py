"""
Model evaluation utilities
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable
import time

from .metrics import Metrics, ClassificationMetrics, RegressionMetrics


class Evaluator:
    """
    Base evaluator class for neural network models
    """

    def __init__(self, model, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator

        Args:
            model: Neural network model
            metrics: List of metric names to compute
        """
        self.model = model
        self.metrics = metrics or ['accuracy']

    def evaluate(self, data_loader, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model on data

        Args:
            data_loader: Data loader
            verbose: Whether to print results

        Returns:
            Dictionary of metric results
        """
        self.model.eval_mode()

        all_predictions = []
        all_targets = []

        start_time = time.time()

        for batch_data, batch_labels in data_loader:
            predictions = self.model.forward(batch_data)
            all_predictions.append(predictions)
            all_targets.append(batch_labels)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        results = self.compute_metrics(all_predictions, all_targets)
        eval_time = time.time() - start_time

        if verbose:
            print(f"\nEvaluation completed in {eval_time:.2f}s")
            for metric_name, value in results.items():
                print(f"{metric_name}: {value:.4f}")

        return results

    def compute_metrics(self, predictions: np.ndarray,
                       targets: np.ndarray) -> Dict[str, float]:
        """
        Compute specified metrics

        Args:
            predictions: Model predictions
            targets: True targets

        Returns:
            Dictionary of metrics
        """
        results = {}

        for metric_name in self.metrics:
            if hasattr(Metrics, metric_name):
                metric_fn = getattr(Metrics, metric_name)
                results[metric_name] = metric_fn(predictions, targets)

        return results


class ModelEvaluator(Evaluator):
    """
    Advanced model evaluator with comprehensive metrics
    """

    def __init__(self, model, task_type: str = 'classification',
                 metrics: Optional[List[str]] = None):
        """
        Initialize model evaluator

        Args:
            model: Neural network model
            task_type: 'classification' or 'regression'
            metrics: List of metric names
        """
        super().__init__(model, metrics)
        self.task_type = task_type

        if metrics is None:
            if task_type == 'classification':
                self.metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            else:
                self.metrics = ['mean_squared_error', 'mean_absolute_error', 'r2_score']

    def compute_metrics(self, predictions: np.ndarray,
                       targets: np.ndarray) -> Dict[str, float]:
        """
        Compute task-specific metrics

        Args:
            predictions: Model predictions
            targets: True targets

        Returns:
            Dictionary of metrics
        """
        results = {}

        if self.task_type == 'classification':
            pred_classes = np.argmax(predictions, axis=-1)
            true_classes = np.argmax(targets, axis=-1) if targets.ndim > 1 else targets

            for metric_name in self.metrics:
                if hasattr(ClassificationMetrics, metric_name):
                    metric_fn = getattr(ClassificationMetrics, metric_name)
                    results[metric_name] = metric_fn(pred_classes, true_classes)

        else:
            for metric_name in self.metrics:
                if hasattr(RegressionMetrics, metric_name):
                    metric_fn = getattr(RegressionMetrics, metric_name)
                    results[metric_name] = metric_fn(predictions, targets)
                elif hasattr(Metrics, metric_name):
                    metric_fn = getattr(Metrics, metric_name)
                    results[metric_name] = metric_fn(predictions, targets)

        return results

    def cross_validate(self, data: np.ndarray, labels: np.ndarray,
                      k_folds: int = 5, shuffle: bool = True) -> Dict[str, List[float]]:
        """
        Perform k-fold cross validation

        Args:
            data: Input data
            labels: Target labels
            k_folds: Number of folds
            shuffle: Whether to shuffle data

        Returns:
            Dictionary of metric lists across folds
        """
        num_samples = len(data)
        indices = np.arange(num_samples)

        if shuffle:
            np.random.shuffle(indices)

        fold_size = num_samples // k_folds
        results = {metric: [] for metric in self.metrics}

        for fold in range(k_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else num_samples

            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

            train_data, train_labels = data[train_indices], labels[train_indices]
            val_data, val_labels = data[val_indices], labels[val_indices]

            self.model.eval_mode()
            predictions = self.model.forward(val_data)

            fold_results = self.compute_metrics(predictions, val_labels)

            for metric_name, value in fold_results.items():
                results[metric_name].append(value)

            print(f"Fold {fold + 1}/{k_folds} completed")

        print("\nCross-validation results:")
        for metric_name, values in results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name}: {mean_val:.4f} (+/- {std_val:.4f})")

        return results

    def compute_confusion_matrix(self, data_loader) -> np.ndarray:
        """
        Compute confusion matrix

        Args:
            data_loader: Data loader

        Returns:
            Confusion matrix
        """
        self.model.eval_mode()

        all_predictions = []
        all_targets = []

        for batch_data, batch_labels in data_loader:
            predictions = self.model.forward(batch_data)
            pred_classes = np.argmax(predictions, axis=-1)
            all_predictions.append(pred_classes)

            if batch_labels.ndim > 1:
                true_classes = np.argmax(batch_labels, axis=-1)
            else:
                true_classes = batch_labels
            all_targets.append(true_classes)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        return ClassificationMetrics.confusion_matrix(all_predictions, all_targets)

    def analyze_errors(self, data: np.ndarray, labels: np.ndarray,
                      num_samples: int = 10) -> Dict[str, Any]:
        """
        Analyze model errors

        Args:
            data: Input data
            labels: True labels
            num_samples: Number of error samples to return

        Returns:
            Dictionary with error analysis
        """
        self.model.eval_mode()
        predictions = self.model.forward(data)

        if self.task_type == 'classification':
            pred_classes = np.argmax(predictions, axis=-1)
            true_classes = np.argmax(labels, axis=-1) if labels.ndim > 1 else labels

            errors = pred_classes != true_classes
            error_indices = np.where(errors)[0]

            error_samples = {
                'num_errors': len(error_indices),
                'error_rate': len(error_indices) / len(data),
                'error_indices': error_indices[:num_samples].tolist(),
                'predictions': pred_classes[error_indices[:num_samples]].tolist(),
                'true_labels': true_classes[error_indices[:num_samples]].tolist()
            }

        else:
            errors = np.abs(predictions - labels)
            worst_indices = np.argsort(errors.flatten())[-num_samples:]

            error_samples = {
                'mean_error': np.mean(errors),
                'max_error': np.max(errors),
                'worst_indices': worst_indices.tolist(),
                'worst_errors': errors[worst_indices].tolist()
            }

        return error_samples

    def benchmark(self, data_loader, num_runs: int = 3) -> Dict[str, float]:
        """
        Benchmark model performance

        Args:
            data_loader: Data loader
            num_runs: Number of benchmark runs

        Returns:
            Benchmark statistics
        """
        self.model.eval_mode()

        inference_times = []
        throughputs = []

        for run in range(num_runs):
            start_time = time.time()
            num_samples = 0

            for batch_data, _ in data_loader:
                _ = self.model.forward(batch_data)
                num_samples += len(batch_data)

            elapsed_time = time.time() - start_time
            inference_times.append(elapsed_time)
            throughputs.append(num_samples / elapsed_time)

        return {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs)
        }
