"""
Feature selection utilities
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE, SelectFromModel
)


class FeatureSelector:
    """Select most important features for ML models"""
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        
    def select_by_variance(
        self, 
        data: pd.DataFrame, 
        threshold: float = 0.01
    ) -> List[str]:
        """Select features based on variance threshold"""
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(data)
        
        selected_features = data.columns[selector.get_support()].tolist()
        self.selected_features = selected_features
        
        return selected_features
    
    def select_by_correlation(
        self, 
        data: pd.DataFrame, 
        target: Union[pd.Series, str],
        threshold: float = 0.5,
        method: str = 'pearson'
    ) -> List[str]:
        """Select features based on correlation with target"""
        if isinstance(target, str):
            target_col = data[target]
            features = data.drop(columns=[target])
        else:
            target_col = target
            features = data
        
        correlations = features.corrwith(target_col, method=method).abs()
        selected_features = correlations[correlations >= threshold].index.tolist()
        
        self.selected_features = selected_features
        self.feature_scores = correlations.to_dict()
        
        return selected_features
    
    def select_k_best(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        k: int = 10,
        score_func: str = 'f_classif'
    ) -> List[str]:
        """Select k best features using statistical tests"""
        score_functions = {
            'f_classif': f_classif,
            'f_regression': f_regression,
            'mutual_info_classif': mutual_info_classif,
            'mutual_info_regression': mutual_info_regression
        }
        
        selector = SelectKBest(score_func=score_functions[score_func], k=k)
        selector.fit(X, y)
        
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_scores = dict(zip(X.columns, selector.scores_))
        else:
            selected_features = list(selector.get_support(indices=True))
            self.feature_scores = dict(enumerate(selector.scores_))
        
        self.selected_features = selected_features
        
        return selected_features
    
    def select_by_model(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        estimator,
        threshold: str = 'mean'
    ) -> List[str]:
        """Select features based on model importance"""
        selector = SelectFromModel(estimator, threshold=threshold)
        selector.fit(X, y)
        
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[selector.get_support()].tolist()
            if hasattr(estimator, 'feature_importances_'):
                self.feature_scores = dict(zip(X.columns, estimator.feature_importances_))
            elif hasattr(estimator, 'coef_'):
                self.feature_scores = dict(zip(X.columns, np.abs(estimator.coef_)))
        else:
            selected_features = list(selector.get_support(indices=True))
        
        self.selected_features = selected_features
        
        return selected_features
    
    def select_by_rfe(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        estimator,
        n_features: int = 10,
        step: int = 1
    ) -> List[str]:
        """Select features using Recursive Feature Elimination"""
        selector = RFE(estimator, n_features_to_select=n_features, step=step)
        selector.fit(X, y)
        
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_scores = dict(zip(X.columns, selector.ranking_))
        else:
            selected_features = list(selector.get_support(indices=True))
            self.feature_scores = dict(enumerate(selector.ranking_))
        
        self.selected_features = selected_features
        
        return selected_features
    
    def remove_multicollinear_features(
        self, 
        data: pd.DataFrame, 
        threshold: float = 0.9
    ) -> List[str]:
        """Remove highly correlated features"""
        corr_matrix = data.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > threshold)
        ]
        
        selected_features = [col for col in data.columns if col not in to_drop]
        self.selected_features = selected_features
        
        return selected_features
    
    def get_feature_importance(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        model_type: str = 'random_forest'
    ) -> pd.DataFrame:
        """Get feature importance scores using tree-based models"""
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if len(np.unique(y)) < 20:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            import xgboost as xgb
            if len(np.unique(y)) < 20:
                model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        
        if isinstance(X, pd.DataFrame):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': range(X.shape[1]),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return importance_df
