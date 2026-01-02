"""
Data transformation techniques using learned representations
"""

import numpy as np
from typing import Optional


class PCATransformer:
    """
    Principal Component Analysis - learns linear transformation from data
    """

    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA transformer

        Args:
            n_components: Number of components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'PCATransformer':
        """
        Learn principal components from data

        Args:
            X: Training data

        Returns:
            Self
        """
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if self.n_components is None:
            self.n_components = X.shape[1]

        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using learned components

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("PCA must be fitted before transform")

        X_centered = X - self.mean_
        return X_centered @ self.components_

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
        Inverse transform using learned components

        Args:
            X: Transformed data

        Returns:
            Reconstructed data
        """
        if not self.fitted:
            raise ValueError("PCA must be fitted before inverse_transform")

        return X @ self.components_.T + self.mean_

    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio"""
        if not self.fitted:
            raise ValueError("PCA must be fitted first")

        total_variance = np.sum(self.explained_variance_)
        return self.explained_variance_ / total_variance


class FeatureSelector:
    """
    Feature selection using learned importance scores
    """

    def __init__(self, n_features: Optional[int] = None, threshold: Optional[float] = None):
        """
        Initialize feature selector

        Args:
            n_features: Number of features to select
            threshold: Importance threshold
        """
        self.n_features = n_features
        self.threshold = threshold
        self.feature_importances_ = None
        self.selected_features_ = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """
        Learn feature importance from data

        Args:
            X: Training data
            y: Target labels

        Returns:
            Self
        """
        self.feature_importances_ = self._compute_importances(X, y)

        if self.n_features is not None:
            indices = np.argsort(self.feature_importances_)[::-1]
            self.selected_features_ = indices[:self.n_features]
        elif self.threshold is not None:
            self.selected_features_ = np.where(
                self.feature_importances_ > self.threshold
            )[0]
        else:
            self.selected_features_ = np.arange(X.shape[1])

        self.fitted = True
        return self

    def _compute_importances(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute feature importances using correlation with target

        Args:
            X: Training data
            y: Target labels

        Returns:
            Feature importance scores
        """
        importances = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            correlation = np.abs(np.corrcoef(X[:, i], y.flatten())[0, 1])
            importances[i] = correlation

        return importances

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by selecting important features

        Args:
            X: Data to transform

        Returns:
            Transformed data with selected features
        """
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted before transform")

        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform
            y: Target labels

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def get_support(self) -> np.ndarray:
        """Get selected feature indices"""
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted first")

        return self.selected_features_


class PolynomialFeatures:
    """
    Generate polynomial and interaction features
    """

    def __init__(self, degree: int = 2, interaction_only: bool = False):
        """
        Initialize polynomial features

        Args:
            degree: Polynomial degree
            interaction_only: Only create interaction features
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """
        Fit polynomial features

        Args:
            X: Training data

        Returns:
            Self
        """
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by creating polynomial features

        Args:
            X: Data to transform

        Returns:
            Transformed data with polynomial features
        """
        n_samples, n_features = X.shape
        features = [np.ones((n_samples, 1)), X]

        if self.degree >= 2:
            if self.interaction_only:
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        features.append((X[:, i] * X[:, j]).reshape(-1, 1))
            else:
                for d in range(2, self.degree + 1):
                    for i in range(n_features):
                        features.append((X[:, i] ** d).reshape(-1, 1))

        return np.hstack(features)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
