"""
Regression model implementations
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_model import BaseModel


class Regressor(BaseModel):
    """Regression model wrapper for various algorithms"""
    
    def __init__(
        self, 
        model_type: str = 'random_forest',
        model_name: str = "regressor",
        **kwargs
    ):
        super().__init__(model_name)
        self.model_type = model_type
        self.model_params = kwargs
        
    def build(self, **kwargs):
        """Build regression model"""
        params = {**self.model_params, **kwargs}
        
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**params)
            
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(**params)
            
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**params)
            
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**params)
            
        elif self.model_type == 'catboost':
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(**params)
            
        elif self.model_type == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(**params)
            
        elif self.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            self.model = Ridge(**params)
            
        elif self.model_type == 'lasso':
            from sklearn.linear_model import Lasso
            self.model = Lasso(**params)
            
        elif self.model_type == 'elastic_net':
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(**params)
            
        elif self.model_type == 'svr':
            from sklearn.svm import SVR
            self.model = SVR(**params)
            
        elif self.model_type == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor(**params)
            
        elif self.model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(**params)
            
        elif self.model_type == 'adaboost':
            from sklearn.ensemble import AdaBoostRegressor
            self.model = AdaBoostRegressor(**params)
            
        elif self.model_type == 'extra_trees':
            from sklearn.ensemble import ExtraTreesRegressor
            self.model = ExtraTreesRegressor(**params)
            
        elif self.model_type == 'huber':
            from sklearn.linear_model import HuberRegressor
            self.model = HuberRegressor(**params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self
    
    def train(self, X_train, y_train, **kwargs):
        """Train the regression model"""
        if self.model is None:
            self.build()
        
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Predict continuous values"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return None
