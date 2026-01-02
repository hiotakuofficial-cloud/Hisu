"""Visualization utilities"""

import numpy as np
from typing import List, Optional, Tuple


class Visualizer:
    """Visualization utilities for ML models"""

    @staticmethod
    def plot_learning_curves(history: dict, figsize: Tuple[int, int] = (12, 4)):
        """Plot training and validation curves"""
        try:
            import matplotlib.pyplot as plt

            metrics = ['loss', 'accuracy']
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Val Loss')

            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Model Loss')
            axes[0].legend()
            axes[0].grid(True)

            if 'train_acc' in history:
                axes[1].plot(history['train_acc'], label='Train Accuracy')
            if 'val_acc' in history:
                axes[1].plot(history['val_acc'], label='Val Accuracy')

            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            return fig

        except ImportError:
            print("Matplotlib not available for plotting")
            return None

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None, figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)

            if class_names is not None:
                tick_marks = np.arange(len(class_names))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_names)
                ax.set_yticklabels(class_names)

            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")

            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()

            return fig

        except ImportError:
            print("Matplotlib not available for plotting")
            return None

    @staticmethod
    def plot_feature_importance(importance: np.ndarray, feature_names: Optional[List[str]] = None,
                                 top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """Plot feature importance"""
        try:
            import matplotlib.pyplot as plt

            indices = np.argsort(importance)[::-1][:top_n]
            sorted_importance = importance[indices]

            if feature_names is not None:
                sorted_names = [feature_names[i] for i in indices]
            else:
                sorted_names = [f'Feature {i}' for i in indices]

            fig, ax = plt.subplots(figsize=figsize)
            ax.barh(range(len(sorted_importance)), sorted_importance)
            ax.set_yticks(range(len(sorted_importance)))
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importance')
            ax.invert_yaxis()
            plt.tight_layout()

            return fig

        except ImportError:
            print("Matplotlib not available for plotting")
            return None

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, figsize: Tuple[int, int] = (8, 6)):
        """Plot ROC curve"""
        try:
            import matplotlib.pyplot as plt

            desc_score_indices = np.argsort(y_scores)[::-1]
            y_scores = y_scores[desc_score_indices]
            y_true = y_true[desc_score_indices]

            tps = np.cumsum(y_true)
            fps = np.cumsum(1 - y_true)

            tps = np.concatenate([[0], tps])
            fps = np.concatenate([[0], fps])

            tpr = tps / tps[-1]
            fpr = fps / fps[-1]

            auc = np.trapz(tpr, fpr)

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            return fig

        except ImportError:
            print("Matplotlib not available for plotting")
            return None

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (10, 6)):
        """Plot true vs predicted values"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(y_true, y_pred, alpha=0.5)
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('True vs Predicted Values')
            ax.grid(True)
            plt.tight_layout()

            return fig

        except ImportError:
            print("Matplotlib not available for plotting")
            return None

    @staticmethod
    def plot_distribution(data: np.ndarray, bins: int = 50, figsize: Tuple[int, int] = (10, 6)):
        """Plot data distribution"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Data Distribution')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            return fig

        except ImportError:
            print("Matplotlib not available for plotting")
            return None
