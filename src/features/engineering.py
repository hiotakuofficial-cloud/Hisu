"""
Feature engineering utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Optional


class FeatureEngineer:
    """Create and transform features"""
    
    def __init__(self):
        self.poly_features = None
        
    def create_polynomial_features(
        self, 
        data: pd.DataFrame, 
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """Create polynomial features"""
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_data = self.poly_features.fit_transform(data)
        
        feature_names = self.poly_features.get_feature_names_out(data.columns)
        return pd.DataFrame(poly_data, columns=feature_names, index=data.index)
    
    def create_interaction_features(
        self, 
        data: pd.DataFrame, 
        columns: List[str]
    ) -> pd.DataFrame:
        """Create interaction features between specified columns"""
        df = data.copy()
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return df
    
    def create_binned_features(
        self, 
        data: pd.DataFrame, 
        column: str,
        bins: int = 5,
        labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create binned features from continuous variables"""
        df = data.copy()
        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
        return df
    
    def create_log_features(
        self, 
        data: pd.DataFrame, 
        columns: List[str]
    ) -> pd.DataFrame:
        """Create log-transformed features"""
        df = data.copy()
        
        for col in columns:
            df[f'{col}_log'] = np.log1p(df[col])
        
        return df
    
    def create_aggregated_features(
        self, 
        data: pd.DataFrame, 
        group_by: str,
        agg_columns: List[str],
        agg_funcs: List[str] = ['mean', 'sum', 'std']
    ) -> pd.DataFrame:
        """Create aggregated features based on grouping"""
        df = data.copy()
        
        for col in agg_columns:
            for func in agg_funcs:
                agg_name = f'{col}_{func}_by_{group_by}'
                df[agg_name] = df.groupby(group_by)[col].transform(func)
        
        return df
    
    def create_time_features(
        self, 
        data: pd.DataFrame, 
        datetime_column: str
    ) -> pd.DataFrame:
        """Extract time-based features from datetime column"""
        df = data.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        df[f'{datetime_column}_year'] = df[datetime_column].dt.year
        df[f'{datetime_column}_month'] = df[datetime_column].dt.month
        df[f'{datetime_column}_day'] = df[datetime_column].dt.day
        df[f'{datetime_column}_dayofweek'] = df[datetime_column].dt.dayofweek
        df[f'{datetime_column}_hour'] = df[datetime_column].dt.hour
        df[f'{datetime_column}_quarter'] = df[datetime_column].dt.quarter
        
        return df
