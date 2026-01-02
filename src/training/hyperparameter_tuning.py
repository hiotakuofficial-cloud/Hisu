"""
Hyperparameter tuning utilities
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, Any, Optional


class HyperparameterTuner:
    """Tune model hyperparameters"""
    
    def __init__(self, model, task: str = 'classification'):
        self.model = model
        self.task = task
        self.best_params = None
        self.best_score = None
        self.search_results = None
        
    def grid_search(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        param_grid: Dict[str, Any],
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1
    ):
        """Perform grid search for hyperparameter tuning"""
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.search_results = grid_search.cv_results_
        self.model = grid_search.best_estimator_
        
        return self
    
    def random_search(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """Perform random search for hyperparameter tuning"""
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.search_results = random_search.cv_results_
        self.model = random_search.best_estimator_
        
        return self
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found"""
        if self.best_params is None:
            raise ValueError("No tuning has been performed yet")
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best score achieved"""
        if self.best_score is None:
            raise ValueError("No tuning has been performed yet")
        return self.best_score
    
    def get_best_model(self):
        """Get best model found"""
        return self.model
