"""
Model evaluation metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any, Optional


class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, task: str = 'classification'):
        self.task = task
        
    def evaluate_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = 'binary'
    ) -> Dict[str, Any]:
        """Evaluate classification model"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    if y_pred_proba.ndim == 2:
                        y_pred_proba = y_pred_proba[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        return metrics
    
    def evaluate_regression(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression model"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        
        return metrics
    
    def get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def get_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ):
        """Get ROC curve data"""
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return roc_curve(y_true, y_pred_proba)
    
    def calculate_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate residuals for regression"""
        return y_true - y_pred
