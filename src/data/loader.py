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
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
