"""
Data preprocessing and scaling utilities.
"""
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class FeatureScaler:
    """Feature scaling wrapper with multiple scaling methods."""

    def __init__(self, method: str = 'standard'):
        """
        Initialize feature scaler.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scaler = None
        self._initialize_scaler()

    def _initialize_scaler(self):
        """Initialize the appropriate scaler based on method."""
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """
        Fit scaler to data.

        Args:
            X: Input features

        Returns:
            Self for method chaining
        """
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Input features

        Returns:
            Scaled features
        """
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Input features

        Returns:
            Scaled features
        """
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.

        Args:
            X: Scaled features

        Returns:
            Original scale features
        """
        return self.scaler.inverse_transform(X)


class CategoricalEncoder:
    """Categorical variable encoding utilities."""

    def __init__(self, method: str = 'label'):
        """
        Initialize encoder.

        Args:
            method: Encoding method ('label' or 'onehot')
        """
        self.method = method
        self.encoder = None
        self._initialize_encoder()

    def _initialize_encoder(self):
        """Initialize the appropriate encoder."""
        if self.method == 'label':
            self.encoder = LabelEncoder()
        elif self.method == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")

    def fit(self, X: np.ndarray) -> 'CategoricalEncoder':
        """Fit encoder to data."""
        if self.method == 'onehot' and X.ndim == 1:
            X = X.reshape(-1, 1)
        self.encoder.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted encoder."""
        if self.method == 'onehot' and X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.encoder.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        if self.method == 'onehot' and X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.encoder.fit_transform(X)


class Normalizer:
    """Data normalization utilities."""

    @staticmethod
    def l2_normalize(X: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        L2 normalization.

        Args:
            X: Input data
            axis: Axis along which to normalize

        Returns:
            L2 normalized data
        """
        norm = np.linalg.norm(X, axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return X / norm

    @staticmethod
    def l1_normalize(X: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        L1 normalization.

        Args:
            X: Input data
            axis: Axis along which to normalize

        Returns:
            L1 normalized data
        """
        norm = np.abs(X).sum(axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return X / norm

    @staticmethod
    def max_normalize(X: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Max normalization.

        Args:
            X: Input data
            axis: Axis along which to normalize

        Returns:
            Max normalized data
        """
        max_val = np.abs(X).max(axis=axis, keepdims=True)
        max_val = np.where(max_val == 0, 1, max_val)
        return X / max_val


class OutlierHandler:
    """Handle outliers in data."""

    @staticmethod
    def clip_outliers(
        X: np.ndarray,
        lower_percentile: float = 1,
        upper_percentile: float = 99
    ) -> np.ndarray:
        """
        Clip outliers based on percentiles.

        Args:
            X: Input data
            lower_percentile: Lower percentile for clipping
            upper_percentile: Upper percentile for clipping

        Returns:
            Data with outliers clipped
        """
        lower_bound = np.percentile(X, lower_percentile, axis=0)
        upper_bound = np.percentile(X, upper_percentile, axis=0)
        return np.clip(X, lower_bound, upper_bound)

    @staticmethod
    def remove_outliers_iqr(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        multiplier: float = 1.5
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Remove outliers using IQR method.

        Args:
            X: Input features
            y: Target labels (optional)
            multiplier: IQR multiplier for outlier detection

        Returns:
            Filtered X and y (if provided)
        """
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)

        if y is not None:
            return X[mask], y[mask]
        return X[mask], None

    @staticmethod
    def remove_outliers_zscore(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Remove outliers using Z-score method.

        Args:
            X: Input features
            y: Target labels (optional)
            threshold: Z-score threshold

        Returns:
            Filtered X and y (if provided)
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        z_scores = np.abs((X - mean) / (std + 1e-8))

        mask = np.all(z_scores < threshold, axis=1)

        if y is not None:
            return X[mask], y[mask]
        return X[mask], None
