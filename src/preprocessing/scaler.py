"""Data scaling and normalization"""

import numpy as np
from typing import Optional


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute mean and std for scaling"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse scale data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return X * self.std_ + self.mean_


class MinMaxScaler:
    """Scale features to a given range (default [0, 1])"""

    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Compute min and max for scaling"""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale data to range"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        X_std = (X - self.min_) / (self.max_ - self.min_ + 1e-8)
        return X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse scale data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        X_std = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return X_std * (self.max_ - self.min_) + self.min_


class RobustScaler:
    """Scale features using statistics that are robust to outliers"""

    def __init__(self):
        self.median_ = None
        self.iqr_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """Compute median and IQR for scaling"""
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ == 0] = 1.0
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return (X - self.median_) / self.iqr_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse scale data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return X * self.iqr_ + self.median_
