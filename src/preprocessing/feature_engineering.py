"""
Feature engineering utilities for ML models.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class FeatureEngineer:
    """Feature engineering utilities."""

    @staticmethod
    def create_polynomial_features(
        X: np.ndarray,
        degree: int = 2,
        include_bias: bool = False
    ) -> np.ndarray:
        """
        Create polynomial features.

        Args:
            X: Input features
            degree: Polynomial degree
            include_bias: Whether to include bias term

        Returns:
            Polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        return poly.fit_transform(X)

    @staticmethod
    def create_interaction_features(
        X: np.ndarray,
        feature_pairs: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Create interaction features for specified feature pairs.

        Args:
            X: Input features
            feature_pairs: List of (i, j) index pairs to create interactions

        Returns:
            Original features with interaction features appended
        """
        interactions = []
        for i, j in feature_pairs:
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))

        if interactions:
            return np.hstack([X] + interactions)
        return X

    @staticmethod
    def create_binned_features(
        X: np.ndarray,
        n_bins: int = 5,
        strategy: str = 'quantile'
    ) -> np.ndarray:
        """
        Create binned features.

        Args:
            X: Input features
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')

        Returns:
            Binned features
        """
        from sklearn.preprocessing import KBinsDiscretizer

        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy=strategy
        )
        return discretizer.fit_transform(X)

    @staticmethod
    def create_lag_features(
        X: np.ndarray,
        lags: List[int]
    ) -> np.ndarray:
        """
        Create lag features for time series data.

        Args:
            X: Input time series features
            lags: List of lag periods

        Returns:
            Features with lag features
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        lag_features = [X]
        for lag in lags:
            lagged = np.roll(X, lag, axis=0)
            lagged[:lag] = 0
            lag_features.append(lagged)

        return np.hstack(lag_features)


class DimensionalityReducer:
    """Dimensionality reduction utilities."""

    def __init__(self, method: str = 'pca', n_components: int = 2):
        """
        Initialize dimensionality reducer.

        Args:
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components to reduce to
        """
        self.method = method
        self.n_components = n_components
        self.reducer = None
        self._initialize_reducer()

    def _initialize_reducer(self):
        """Initialize the appropriate reducer."""
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
        elif self.method == 'tsne':
            from sklearn.manifold import TSNE
            self.reducer = TSNE(n_components=self.n_components, random_state=42)
        elif self.method == 'umap':
            try:
                import umap
                self.reducer = umap.UMAP(n_components=self.n_components, random_state=42)
            except ImportError:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")

    def fit(self, X: np.ndarray) -> 'DimensionalityReducer':
        """Fit reducer to data."""
        if self.method == 'tsne':
            raise ValueError("TSNE does not support separate fit/transform")
        self.reducer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted reducer."""
        if self.method == 'tsne':
            raise ValueError("TSNE does not support separate fit/transform. Use fit_transform.")
        return self.reducer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.reducer.fit_transform(X)


class FeatureSelector:
    """Feature selection utilities."""

    def __init__(
        self,
        method: str = 'kbest',
        n_features: int = 10,
        score_func=f_classif
    ):
        """
        Initialize feature selector.

        Args:
            method: Selection method
            n_features: Number of features to select
            score_func: Scoring function for selection
        """
        self.method = method
        self.n_features = n_features
        self.score_func = score_func
        self.selector = None
        self.selected_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """
        Fit feature selector.

        Args:
            X: Input features
            y: Target labels

        Returns:
            Self for method chaining
        """
        if self.method == 'kbest':
            self.selector = SelectKBest(score_func=self.score_func, k=self.n_features)
            self.selector.fit(X, y)
            self.selected_features = self.selector.get_support(indices=True)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
            self.selector.fit(X, y)
            self.selected_features = self.selector.get_support(indices=True)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by selecting features."""
        return self.selector.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        return self.selected_features
