"""
Hyperparameter tuning utilities
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score


class HyperparameterTuner:
    """Tune hyperparameters for ML models"""
    
    def __init__(self, model, task: str = 'classification'):
        self.model = model
        self.task = task
        self.best_params = None
        self.best_score = None
        self.search_results = None
        
    def grid_search(
        self,
        X_train,
        y_train,
        param_grid: Dict[str, list],
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """Perform grid search for hyperparameter tuning"""
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.search_results = grid_search.cv_results_
        self.model = grid_search.best_estimator_
        
        return self
    
    def random_search(
        self,
        X_train,
        y_train,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42
    ):
        """Perform random search for hyperparameter tuning"""
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.search_results = random_search.cv_results_
        self.model = random_search.best_estimator_
        
        return self
    
    def bayesian_optimization(
        self,
        X_train,
        y_train,
        param_space: Dict[str, tuple],
        n_iterations: int = 50,
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: int = 42
    ):
        """Perform Bayesian optimization for hyperparameter tuning"""
        from skopt import BayesSearchCV
        
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        bayes_search = BayesSearchCV(
            self.model,
            param_space,
            n_iter=n_iterations,
            cv=cv,
            scoring=scoring,
            random_state=random_state
        )
        
        bayes_search.fit(X_train, y_train)
        
        self.best_params = bayes_search.best_params_
        self.best_score = bayes_search.best_score_
        self.search_results = bayes_search.cv_results_
        self.model = bayes_search.best_estimator_
        
        return self
    
    def optuna_optimization(
        self,
        X_train,
        y_train,
        param_space_func,
        n_trials: int = 100,
        cv: int = 5,
        scoring: Optional[str] = None
    ):
        """Perform Optuna optimization for hyperparameter tuning"""
        import optuna
        from sklearn.model_selection import cross_val_score
        
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        def objective(trial):
            params = param_space_func(trial)
            
            if hasattr(self.model, 'set_params'):
                self.model.set_params(**params)
            
            scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize' if self.task == 'classification' else 'minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**self.best_params)
        
        self.model.fit(X_train, y_train)
        
        return self
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found"""
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best score achieved"""
        return self.best_score
    
    def get_search_results(self):
        """Get detailed search results"""
        return self.search_results
