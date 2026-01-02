"""
Visualization utilities for ML results and analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
from pathlib import Path


class TrainingVisualizer:
    """Visualize training progress and metrics."""

    @staticmethod
    def plot_learning_curves(
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = 'Learning Curves',
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot learning curves.

        Args:
            train_losses: Training losses
            val_losses: Validation losses
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, List[float]],
        title: str = 'Metrics Over Time',
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot multiple metrics over time.

        Args:
            metrics_dict: Dictionary of metric names to values
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))

        for metric_name, values in metrics_dict.items():
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=metric_name, linewidth=2, marker='o')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


class ModelVisualizer:
    """Visualize model predictions and performance."""

    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = 'Confusion Matrix',
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else range(len(confusion_matrix)),
            yticklabels=class_names if class_names else range(len(confusion_matrix))
        )

        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_predictions_vs_actual(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Predictions vs Actual',
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot predictions against actual values (regression).

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))

        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Residual Plot',
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot residuals for regression analysis.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot
        """
        residuals = y_true - y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: str = 'Feature Importance',
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_names: Names of features
            importances: Importance scores
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save plot
        """
        indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


class DataVisualizer:
    """Visualize data distributions and characteristics."""

    @staticmethod
    def plot_feature_distributions(
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_cols: int = 4,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot distributions of features.

        Args:
            X: Feature data
            feature_names: Names of features
            n_cols: Number of columns in subplot grid
            save_path: Path to save plot
        """
        n_features = X.shape[1]
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i in range(n_features):
            ax = axes[i]
            ax.hist(X[:, i], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)

            if feature_names:
                ax.set_title(feature_names[i], fontsize=11)
            else:
                ax.set_title(f'Feature {i}', fontsize=11)
            ax.grid(True, alpha=0.3)

        for i in range(n_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_correlation_matrix(
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = 'Feature Correlation Matrix',
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot correlation matrix of features.

        Args:
            X: Feature data
            feature_names: Names of features
            title: Plot title
            save_path: Path to save plot
        """
        correlation_matrix = np.corrcoef(X.T)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            xticklabels=feature_names if feature_names else range(X.shape[1]),
            yticklabels=feature_names if feature_names else range(X.shape[1])
        )
        plt.title(title, fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
