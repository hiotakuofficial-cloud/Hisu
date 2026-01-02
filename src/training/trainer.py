"""
Model training utilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from typing import Dict, Any, Optional, List
import joblib


class ModelTrainer:
    """Train and manage machine learning models"""
    
    def __init__(self, model, task: str = 'classification'):
        self.model = model
        self.task = task
        self.training_history = []
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        **kwargs
    ):
        """Train the model"""
        self.model.fit(X_train, y_train, **kwargs)
        return self
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation"""
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
    
    def cross_validate_multiple_metrics(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation with multiple metrics"""
        if scoring is None:
            if self.task == 'classification':
                scoring = ['accuracy', 'precision', 'recall', 'f1']
            else:
                scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = cross_validate(
            self.model, X, y, 
            cv=cv, 
            scoring=scoring,
            return_train_score=True
        )
        
        return cv_results
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = joblib.load(filepath)
        return self
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.model.get_params()
    
    def update_training_history(self, metrics: Dict[str, Any]):
        """Update training history"""
        self.training_history.append(metrics)
