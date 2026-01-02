"""
Data loading utilities for various data sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional


class DataLoader:
    """Load data from various sources"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else None
        
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file"""
        return pd.read_csv(filepath, **kwargs)
    
    def load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from JSON file"""
        return pd.read_json(filepath, **kwargs)
    
    def load_excel(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from Excel file"""
        return pd.read_excel(filepath, **kwargs)
    
    def load_parquet(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from Parquet file"""
        return pd.read_parquet(filepath, **kwargs)
    
    def load_numpy(self, filepath: str) -> np.ndarray:
        """Load data from numpy file"""
        return np.load(filepath)
    
    def split_data(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple:
        """Split data into train, validation, and test sets"""
        from sklearn.model_selection import train_test_split
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        return X_temp, X_test, y_temp, y_test
