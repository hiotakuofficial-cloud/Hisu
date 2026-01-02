"""
Classification model implementations
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_model import BaseModel


class Classifier(BaseModel):
    """Classification model wrapper for various algorithms"""
    
    def __init__(
        self, 
        model_type: str = 'random_forest',
        model_name: str = "classifier",
        **kwargs
    ):
        super().__init__(model_name)
        self.model_type = model_type
        self.model_params = kwargs
        
    def build(self, **kwargs):
        """Build classification model"""
        params = {**self.model_params, **kwargs}
        
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**params)
            
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(**params)
            
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**params)
            
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(**params)
            
        elif self.model_type == 'catboost':
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(**params)
            
        elif self.model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(**params)
            
        elif self.model_type == 'svm':
            from sklearn.svm import SVC
            self.model = SVC(**params)
            
        elif self.model_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(**params)
            
        elif self.model_type == 'naive_bayes':
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB(**params)
            
        elif self.model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(**params)
            
        elif self.model_type == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier
            self.model = AdaBoostClassifier(**params)
            
        elif self.model_type == 'extra_trees':
            from sklearn.ensemble import ExtraTreesClassifier
            self.model = ExtraTreesClassifier(**params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self
    
    def train(self, X_train, y_train, **kwargs):
        """Train the classification model"""
        if self.model is None:
            self.build()
        
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_type} does not support probability predictions")
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None
