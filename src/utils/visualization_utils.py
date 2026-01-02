"""
Visualization utilities for neural network training and evaluation
"""

import numpy as np
from typing import List, Dict, Any, Optional


def plot_learning_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot learning curves from training history

    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
    """
    print("\nLearning Curves:")
    print("-" * 50)

    if 'train_loss' in history:
        print(f"Training Loss: {history['train_loss'][-5:]}")

    if 'val_loss' in history:
        print(f"Validation Loss: {history['val_loss'][-5:]}")

    if 'train_metrics' in history and history['train_metrics']:
        print(f"Training Metrics: {history['train_metrics'][-5:]}")

    if 'val_metrics' in history and history['val_metrics']:
        print(f"Validation Metrics: {history['val_metrics'][-5:]}")

    print("-" * 50)


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save plot
    """
    print("\nConfusion Matrix:")
    print("-" * 50)

    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    header = "True\\Pred  " + "  ".join([f"{name:>8}" for name in class_names])
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>10}  " + "  ".join([f"{val:>8}" for val in row])
        print(row_str)

    print("-" * 50)


def plot_metrics(metrics: Dict[str, float]):
    """
    Display metrics in formatted way

    Args:
        metrics: Dictionary of metrics
    """
    print("\nMetrics:")
    print("-" * 50)

    for metric_name, value in metrics.items():
        print(f"{metric_name:>20}: {value:.4f}")

    print("-" * 50)


def plot_predictions(predictions: np.ndarray, targets: np.ndarray,
                    num_samples: int = 10):
    """
    Display predictions vs targets

    Args:
        predictions: Model predictions
        targets: True targets
        num_samples: Number of samples to display
    """
    print("\nPredictions vs Targets:")
    print("-" * 50)
    print(f"{'Index':>6}  {'Prediction':>12}  {'Target':>12}  {'Error':>12}")
    print("-" * 50)

    for i in range(min(num_samples, len(predictions))):
        pred = predictions[i]
        target = targets[i]
        error = np.abs(pred - target)

        if pred.size > 1:
            pred_val = np.argmax(pred)
            target_val = np.argmax(target) if target.size > 1 else target
        else:
            pred_val = pred
            target_val = target

        print(f"{i:>6}  {pred_val:>12}  {target_val:>12}  {error:>12.4f}")

    print("-" * 50)


def plot_training_progress(epoch: int, total_epochs: int, train_loss: float,
                          val_loss: Optional[float] = None,
                          metrics: Optional[Dict[str, float]] = None):
    """
    Display training progress

    Args:
        epoch: Current epoch
        total_epochs: Total epochs
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Additional metrics
    """
    progress = (epoch + 1) / total_epochs
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '=' * filled + '-' * (bar_length - filled)

    print(f"\rEpoch [{epoch + 1}/{total_epochs}] [{bar}] ", end='')
    print(f"train_loss: {train_loss:.4f}", end='')

    if val_loss is not None:
        print(f" - val_loss: {val_loss:.4f}", end='')

    if metrics:
        for name, value in metrics.items():
            print(f" - {name}: {value:.4f}", end='')

    print()


def generate_summary_report(model_info: Dict[str, Any], training_history: Dict[str, List[float]],
                          final_metrics: Dict[str, float]) -> str:
    """
    Generate summary report

    Args:
        model_info: Model information
        training_history: Training history
        final_metrics: Final evaluation metrics

    Returns:
        Summary report string
    """
    report = []
    report.append("=" * 70)
    report.append("MODEL SUMMARY REPORT")
    report.append("=" * 70)
    report.append("")

    report.append("Model Architecture:")
    report.append("-" * 70)
    for key, value in model_info.items():
        report.append(f"  {key}: {value}")
    report.append("")

    report.append("Training History:")
    report.append("-" * 70)
    if 'train_loss' in training_history:
        final_train_loss = training_history['train_loss'][-1]
        report.append(f"  Final Training Loss: {final_train_loss:.4f}")
    if 'val_loss' in training_history:
        final_val_loss = training_history['val_loss'][-1]
        report.append(f"  Final Validation Loss: {final_val_loss:.4f}")
    report.append(f"  Total Epochs: {len(training_history.get('train_loss', []))}")
    report.append("")

    report.append("Final Evaluation Metrics:")
    report.append("-" * 70)
    for metric_name, value in final_metrics.items():
        report.append(f"  {metric_name}: {value:.4f}")
    report.append("")

    report.append("=" * 70)

    return "\n".join(report)
