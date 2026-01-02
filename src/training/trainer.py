"""
Model training utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.model_selection import cross_val_score, cross_validate


class ModelTrainer:
    """Train and manage ML models"""
    
    def __init__(self, model, task: str = 'classification'):
        self.model = model
        self.task = task
        self.training_history = {}
        
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs
    ):
        """Train the model"""
        if hasattr(self.model, 'train'):
            self.model.train(X_train, y_train, **kwargs)
        else:
            self.model.fit(X_train, y_train, **kwargs)
        
        return self
    
    def cross_validate(
        self,
        X,
        y,
        cv: int = 5,
        scoring: Optional[str] = None,
        return_train_score: bool = True
    ) -> Dict[str, np.ndarray]:
        """Perform cross-validation"""
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score
        )
        
        return cv_results
    
    def train_with_early_stopping(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        patience: int = 10,
        min_delta: float = 0.001,
        **kwargs
    ):
        """Train with early stopping based on validation performance"""
        best_score = float('-inf') if self.task == 'classification' else float('inf')
        patience_counter = 0
        
        for epoch in range(kwargs.get('epochs', 100)):
            self.model.fit(X_train, y_train)
            
            val_predictions = self.model.predict(X_val)
            
            if self.task == 'classification':
                from sklearn.metrics import accuracy_score
                val_score = accuracy_score(y_val, val_predictions)
                improved = val_score > best_score + min_delta
            else:
                from sklearn.metrics import mean_squared_error
                val_score = mean_squared_error(y_val, val_predictions)
                improved = val_score < best_score - min_delta
            
            if improved:
                best_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return self
    
    def ensemble_train(
        self,
        models: List,
        X_train,
        y_train,
        method: str = 'voting'
    ):
        """Train ensemble of models"""
        if method == 'voting':
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            if self.task == 'classification':
                ensemble = VotingClassifier(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
                    voting='soft'
                )
            else:
                ensemble = VotingRegressor(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(models)]
                )
        
        elif method == 'stacking':
            from sklearn.ensemble import StackingClassifier, StackingRegressor
            
            if self.task == 'classification':
                ensemble = StackingClassifier(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(models)]
                )
            else:
                ensemble = StackingRegressor(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(models)]
                )
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        ensemble.fit(X_train, y_train)
        self.model = ensemble
        
        return self
    
    def incremental_train(
        self,
        X_batches: List,
        y_batches: List
    ):
        """Train model incrementally on batches"""
        if not hasattr(self.model, 'partial_fit'):
            raise ValueError("Model does not support incremental learning")
        
        for X_batch, y_batch in zip(X_batches, y_batches):
            self.model.partial_fit(X_batch, y_batch)
        
        return self
    
    def save_training_history(self, filepath: str):
        """Save training history to file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_training_history(self, filepath: str):
        """Load training history from file"""
        import json
        
        with open(filepath, 'r') as f:
            self.training_history = json.load(f)
