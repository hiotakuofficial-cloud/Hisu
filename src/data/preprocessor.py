"""
Data preprocessing and transformation utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Union


class DataPreprocessor:
    """Preprocess and transform data for ML models"""
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        strategy: str = 'mean',
        fill_value: Optional[Union[str, int, float]] = None
    ) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        if strategy == 'drop':
            return data.dropna()
        
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
        
        return data
    
    def scale_features(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        method: str = 'standard',
        fit: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Scale numerical features"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
        
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        return scaled_data
    
    def encode_categorical(
        self, 
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """Encode categorical variables"""
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'onehot':
            return pd.get_dummies(data, columns=columns, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            data_copy = data.copy()
            for col in columns:
                le = LabelEncoder()
                data_copy[col] = le.fit_transform(data_copy[col].astype(str))
            return data_copy
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def remove_outliers(
        self, 
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """Remove outliers from the dataset"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_clean = data.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                data_clean = data_clean[
                    (data_clean[col] >= lower_bound) & 
                    (data_clean[col] <= upper_bound)
                ]
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(data_clean[col]))
                data_clean = data_clean[z_scores < threshold]
        
        return data_clean
    
    def balance_dataset(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        method: str = 'smote'
    ):
        """Balance imbalanced datasets"""
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=42)
        elif method == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        return sampler.fit_resample(X, y)
