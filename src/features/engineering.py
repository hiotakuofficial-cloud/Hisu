"""
Feature engineering utilities
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Callable


class FeatureEngineer:
    """Create and transform features for ML models"""
    
    def __init__(self):
        self.created_features = []
        
    def create_polynomial_features(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(data[columns])
        
        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
        
        result = pd.concat([data, poly_df], axis=1)
        self.created_features.extend(feature_names.tolist())
        
        return result
    
    def create_interaction_features(
        self, 
        data: pd.DataFrame, 
        column_pairs: List[tuple]
    ) -> pd.DataFrame:
        """Create interaction features between column pairs"""
        result = data.copy()
        
        for col1, col2 in column_pairs:
            feature_name = f"{col1}_x_{col2}"
            result[feature_name] = data[col1] * data[col2]
            self.created_features.append(feature_name)
        
        return result
    
    def create_aggregation_features(
        self, 
        data: pd.DataFrame, 
        group_by: str,
        agg_columns: List[str],
        agg_funcs: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """Create aggregation features based on grouping"""
        result = data.copy()
        
        for col in agg_columns:
            for func in agg_funcs:
                feature_name = f"{col}_{func}_by_{group_by}"
                agg_values = data.groupby(group_by)[col].transform(func)
                result[feature_name] = agg_values
                self.created_features.append(feature_name)
        
        return result
    
    def create_time_features(
        self, 
        data: pd.DataFrame, 
        datetime_column: str
    ) -> pd.DataFrame:
        """Extract time-based features from datetime column"""
        result = data.copy()
        dt_col = pd.to_datetime(result[datetime_column])
        
        time_features = {
            f"{datetime_column}_year": dt_col.dt.year,
            f"{datetime_column}_month": dt_col.dt.month,
            f"{datetime_column}_day": dt_col.dt.day,
            f"{datetime_column}_dayofweek": dt_col.dt.dayofweek,
            f"{datetime_column}_hour": dt_col.dt.hour,
            f"{datetime_column}_quarter": dt_col.dt.quarter,
            f"{datetime_column}_is_weekend": dt_col.dt.dayofweek.isin([5, 6]).astype(int)
        }
        
        for feature_name, feature_values in time_features.items():
            result[feature_name] = feature_values
            self.created_features.append(feature_name)
        
        return result
    
    def create_binned_features(
        self, 
        data: pd.DataFrame, 
        column: str,
        bins: int = 5,
        labels: Optional[List] = None
    ) -> pd.DataFrame:
        """Create binned categorical features from continuous variables"""
        result = data.copy()
        feature_name = f"{column}_binned"
        
        result[feature_name] = pd.cut(data[column], bins=bins, labels=labels)
        self.created_features.append(feature_name)
        
        return result
    
    def create_lag_features(
        self, 
        data: pd.DataFrame, 
        column: str,
        lags: List[int] = [1, 2, 3]
    ) -> pd.DataFrame:
        """Create lag features for time series data"""
        result = data.copy()
        
        for lag in lags:
            feature_name = f"{column}_lag_{lag}"
            result[feature_name] = data[column].shift(lag)
            self.created_features.append(feature_name)
        
        return result
    
    def create_rolling_features(
        self, 
        data: pd.DataFrame, 
        column: str,
        windows: List[int] = [3, 7, 14],
        agg_funcs: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """Create rolling window features for time series data"""
        result = data.copy()
        
        for window in windows:
            for func in agg_funcs:
                feature_name = f"{column}_rolling_{window}_{func}"
                result[feature_name] = data[column].rolling(window=window).agg(func)
                self.created_features.append(feature_name)
        
        return result
    
    def apply_custom_transform(
        self, 
        data: pd.DataFrame, 
        column: str,
        transform_func: Callable,
        feature_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Apply custom transformation function to create new feature"""
        result = data.copy()
        
        if feature_name is None:
            feature_name = f"{column}_transformed"
        
        result[feature_name] = data[column].apply(transform_func)
        self.created_features.append(feature_name)
        
        return result
