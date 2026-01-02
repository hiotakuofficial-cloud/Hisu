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
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            try:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                pass
        
        return results
    
    def evaluate_regression(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression model"""
        results = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        
        return results
    
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
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        return fpr, tpr, thresholds
    
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate model based on task type"""
        if self.task == 'classification':
            return self.evaluate_classification(y_true, y_pred, y_pred_proba)
        else:
            return self.evaluate_regression(y_true, y_pred)
