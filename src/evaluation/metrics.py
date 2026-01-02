"""
Model evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)


class ModelEvaluator:
    """Evaluate ML model performance"""
    
    def __init__(self, task: str = 'classification'):
        self.task = task
        self.metrics = {}
        
    def evaluate_classification(
        self,
        y_true,
        y_pred,
        y_pred_proba=None,
        average: str = 'weighted'
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
                    if y_pred_proba.ndim > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
            except Exception as e:
                metrics['roc_auc'] = None
        
        metrics['classification_report'] = classification_report(y_true, y_pred)
        
        self.metrics = metrics
        return metrics
    
    def evaluate_regression(
        self,
        y_true,
        y_pred
    ) -> Dict[str, float]:
        """Evaluate regression model"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        self.metrics = metrics
        return metrics
    
    def evaluate(
        self,
        y_true,
        y_pred,
        y_pred_proba=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate model based on task type"""
        if self.task == 'classification':
            return self.evaluate_classification(y_true, y_pred, y_pred_proba, **kwargs)
        else:
            return self.evaluate_regression(y_true, y_pred)
    
    def cross_validation_scores(
        self,
        model,
        X,
        y,
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate cross-validation scores"""
        from sklearn.model_selection import cross_val_score, cross_validate
        
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'] if self.task == 'classification' else ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        results = {}
        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                metric_name = metric.replace('test_', '')
                results[f'{metric_name}_mean'] = scores.mean()
                results[f'{metric_name}_std'] = scores.std()
        
        return results
    
    def calculate_feature_importance(
        self,
        model,
        feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """Calculate and return feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances[0]
        else:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def learning_curve_analysis(
        self,
        model,
        X,
        y,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        cv: int = 5
    ):
        """Analyze learning curves"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1
        )
        
        return {
            'train_sizes': train_sizes,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stored metrics"""
        return self.metrics
    
    def print_metrics(self):
        """Print metrics in readable format"""
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        
        for metric, value in self.metrics.items():
            if metric == 'confusion_matrix':
                print(f"\n{metric.upper()}:")
                print(value)
            elif metric == 'classification_report':
                print(f"\n{metric.upper()}:")
                print(value)
            elif isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        print("="*50 + "\n")
