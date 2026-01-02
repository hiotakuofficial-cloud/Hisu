"""
Model visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Optional, List


class ModelVisualizer:
    """Visualize model performance and results"""
    
    def __init__(self):
        sns.set_style('whitegrid')
        
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        figsize: tuple = (8, 6),
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        figsize: tuple = (8, 6),
        save_path: Optional[str] = None
    ):
        """Plot ROC curve"""
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(
        self, 
        feature_names: List[str], 
        importances: np.ndarray,
        top_n: int = 20,
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None
    ):
        """Plot feature importance"""
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=figsize)
        plt.title(f'Top {top_n} Feature Importances')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(
        self, 
        history: dict,
        metrics: List[str] = ['loss'],
        figsize: tuple = (12, 4),
        save_path: Optional[str] = None
    ):
        """Plot training history for neural networks"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in history:
                axes[idx].plot(history[metric], label=f'Train {metric}')
                if f'val_{metric}' in history:
                    axes[idx].plot(history[f'val_{metric}'], label=f'Val {metric}')
                axes[idx].set_title(f'{metric.capitalize()} over Epochs')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric.capitalize())
                axes[idx].legend()
                axes[idx].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions_vs_actual(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        figsize: tuple = (8, 6),
        save_path: Optional[str] = None
    ):
        """Plot predictions vs actual values for regression"""
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        figsize: tuple = (12, 4),
        save_path: Optional[str] = None
    ):
        """Plot residuals for regression"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        axes[0].grid(True)
        
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
