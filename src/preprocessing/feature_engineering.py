"""Feature engineering and extraction"""

import numpy as np
from typing import List, Optional


class FeatureEngineering:
    """Feature engineering techniques"""

    @staticmethod
    def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
        """Generate polynomial features"""
        n_samples, n_features = X.shape
        features = [X]

        for d in range(2, degree + 1):
            for i in range(n_features):
                features.append((X[:, i] ** d).reshape(-1, 1))

        for i in range(n_features):
            for j in range(i + 1, n_features):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.concatenate(features, axis=1)

    @staticmethod
    def interaction_features(X: np.ndarray, feature_pairs: Optional[List[tuple]] = None) -> np.ndarray:
        """Generate interaction features"""
        n_samples, n_features = X.shape

        if feature_pairs is None:
            feature_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

        interactions = []
        for i, j in feature_pairs:
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.concatenate([X] + interactions, axis=1)

    @staticmethod
    def binning(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Bin continuous features"""
        binned = np.zeros_like(X)

        for i in range(X.shape[1]):
            bins = np.linspace(X[:, i].min(), X[:, i].max(), n_bins + 1)
            binned[:, i] = np.digitize(X[:, i], bins) - 1

        return binned

    @staticmethod
    def log_transform(X: np.ndarray, offset: float = 1.0) -> np.ndarray:
        """Apply log transformation"""
        return np.log(X + offset)

    @staticmethod
    def sqrt_transform(X: np.ndarray) -> np.ndarray:
        """Apply square root transformation"""
        return np.sqrt(np.abs(X)) * np.sign(X)

    @staticmethod
    def power_transform(X: np.ndarray, power: float = 0.5) -> np.ndarray:
        """Apply power transformation"""
        return np.power(np.abs(X), power) * np.sign(X)

    @staticmethod
    def rolling_statistics(X: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Compute rolling statistics for time series"""
        n_samples, n_features = X.shape
        features = []

        for i in range(n_features):
            rolling_mean = np.zeros(n_samples)
            rolling_std = np.zeros(n_samples)

            for j in range(n_samples):
                start = max(0, j - window_size + 1)
                window = X[start:j + 1, i]

                rolling_mean[j] = np.mean(window)
                rolling_std[j] = np.std(window)

            features.append(rolling_mean.reshape(-1, 1))
            features.append(rolling_std.reshape(-1, 1))

        return np.concatenate([X] + features, axis=1)

    @staticmethod
    def lag_features(X: np.ndarray, n_lags: int = 5) -> np.ndarray:
        """Create lag features for time series"""
        n_samples, n_features = X.shape
        features = [X]

        for lag in range(1, n_lags + 1):
            lagged = np.zeros_like(X)
            lagged[lag:] = X[:-lag]
            features.append(lagged)

        return np.concatenate(features, axis=1)

    @staticmethod
    def difference_features(X: np.ndarray, n_differences: int = 1) -> np.ndarray:
        """Create difference features for time series"""
        features = [X]

        for d in range(1, n_differences + 1):
            diff = np.zeros_like(X)
            diff[1:] = np.diff(X, axis=0)
            features.append(diff)

        return np.concatenate(features, axis=1)
