"""
Data scaling and normalization using learned parameters
"""

import numpy as np
from typing import Optional


class StandardScaler:
    """
    Standardize features by learning mean and std from training data
    """

    def __init__(self):
        """Initialize scaler"""
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Learn mean and std from data

        Args:
            X: Training data

        Returns:
            Self
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using learned parameters

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        return (X - self.mean_) / (self.std_ + 1e-8)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform using learned parameters

        Args:
            X: Transformed data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features by learning min and max from training data
    """

    def __init__(self, feature_range: tuple = (0, 1)):
        """
        Initialize scaler

        Args:
            feature_range: Target range for scaling
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Learn min and max from data

        Args:
            X: Training data

        Returns:
            Self
        """
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using learned parameters

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        X_std = (X - self.min_) / (self.max_ - self.min_ + 1e-8)
        min_val, max_val = self.feature_range
        return X_std * (max_val - min_val) + min_val

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform using learned parameters

        Args:
            X: Transformed data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        min_val, max_val = self.feature_range
        X_std = (X - min_val) / (max_val - min_val)
        return X_std * (self.max_ - self.min_) + self.min_


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers
    Learns median and IQR from training data
    """

    def __init__(self):
        """Initialize scaler"""
        self.median_ = None
        self.iqr_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """
        Learn median and IQR from data

        Args:
            X: Training data

        Returns:
            Self
        """
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using learned parameters

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        return (X - self.median_) / (self.iqr_ + 1e-8)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform using learned parameters

        Args:
            X: Transformed data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        return X * self.iqr_ + self.median_


class Normalizer:
    """
    Normalize samples individually to unit norm
    """

    def __init__(self, norm: str = 'l2'):
        """
        Initialize normalizer

        Args:
            norm: Norm type ('l1', 'l2', 'max')
        """
        self.norm = norm

    def fit(self, X: np.ndarray) -> 'Normalizer':
        """
        Normalizer doesn't need fitting

        Args:
            X: Data (unused)

        Returns:
            Self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize data

        Args:
            X: Data to normalize

        Returns:
            Normalized data
        """
        if self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        return X / (norms + 1e-8)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform

        Returns:
            Normalized data
        """
        return self.fit(X).transform(X)
