"""
Data preprocessing and cleaning utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Optional, List, Union


class DataPreprocessor:
    """Preprocess and clean data"""
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Handle missing values in dataset"""
        df = data.copy()
        cols = columns if columns else df.columns
        
        self.imputer = SimpleImputer(strategy=strategy)
        df[cols] = self.imputer.fit_transform(df[cols])
        
        return df
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return data.drop_duplicates()
    
    def remove_outliers(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method"""
        df = data.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        return df
    
    def scale_features(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        method: str = 'standard'
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Scale features using StandardScaler or MinMaxScaler"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        if isinstance(data, pd.DataFrame):
            scaled_data = self.scaler.fit_transform(data)
            return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        else:
            return self.scaler.fit_transform(data)
    
    def encode_categorical(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        method: str = 'label'
    ) -> pd.DataFrame:
        """Encode categorical variables"""
        df = data.copy()
        
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        
        return df
    
    def normalize_text(self, text: str) -> str:
        """Normalize text data"""
        text = text.lower().strip()
        return text
