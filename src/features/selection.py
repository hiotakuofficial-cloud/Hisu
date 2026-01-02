"""
Feature selection utilities
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Union, Optional


class FeatureSelector:
    """Select most important features"""
    
    def __init__(self):
        self.selector = None
        self.selected_features = None
        
    def select_k_best(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        k: int = 10,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """Select K best features using statistical tests"""
        if task == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
        
        self.selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def select_by_mutual_info(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        k: int = 10,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """Select features using mutual information"""
        if task == 'classification':
            score_func = mutual_info_classif
        else:
            score_func = mutual_info_regression
        
        self.selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def select_by_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        threshold: float = 0.01,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """Select features based on tree-based model importance"""
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        importances = model.feature_importances_
        
        self.selected_features = X.columns[importances > threshold].tolist()
        return X[self.selected_features]
    
    def recursive_feature_elimination(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_features: int = 10,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """Select features using Recursive Feature Elimination"""
        if task == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        self.selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = self.selector.fit_transform(X, y)
        
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def get_feature_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """Get feature importance scores"""
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
