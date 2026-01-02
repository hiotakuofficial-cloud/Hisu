"""
Encoding utilities for categorical features using learned mappings
"""

import numpy as np
from typing import Dict, List, Optional


class LabelEncoder:
    """
    Encode categorical labels as integers by learning mappings
    """

    def __init__(self):
        """Initialize label encoder"""
        self.classes_ = None
        self.class_to_idx_ = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """
        Learn label mappings from data

        Args:
            y: Labels to encode

        Returns:
            Self
        """
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels using learned mappings

        Args:
            y: Labels to transform

        Returns:
            Encoded labels
        """
        if not self.fitted:
            raise ValueError("LabelEncoder must be fitted before transform")

        encoded = np.array([self.class_to_idx_[label] for label in y])
        return encoded

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform labels

        Args:
            y: Labels to fit and transform

        Returns:
            Encoded labels
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform using learned mappings

        Args:
            y: Encoded labels

        Returns:
            Original labels
        """
        if not self.fitted:
            raise ValueError("LabelEncoder must be fitted before inverse_transform")

        return np.array([self.classes_[idx] for idx in y])


class OneHotEncoder:
    """
    One-hot encode categorical features by learning categories
    """

    def __init__(self, sparse: bool = False):
        """
        Initialize one-hot encoder

        Args:
            sparse: Whether to return sparse matrix
        """
        self.sparse = sparse
        self.categories_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Learn categories from data

        Args:
            X: Data to fit

        Returns:
            Self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            self.categories_.append(unique_values)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using learned categories

        Args:
            X: Data to transform

        Returns:
            One-hot encoded data
        """
        if not self.fitted:
            raise ValueError("OneHotEncoder must be fitted before transform")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded_features = []

        for i in range(X.shape[1]):
            categories = self.categories_[i]
            n_categories = len(categories)
            category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

            encoded = np.zeros((X.shape[0], n_categories))
            for j, value in enumerate(X[:, i]):
                if value in category_to_idx:
                    encoded[j, category_to_idx[value]] = 1

            encoded_features.append(encoded)

        return np.hstack(encoded_features)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform

        Returns:
            One-hot encoded data
        """
        return self.fit(X).transform(X)


class FeatureEncoder:
    """
    Encode features using learned embeddings (for neural networks)
    """

    def __init__(self, embedding_dim: int = 32):
        """
        Initialize feature encoder

        Args:
            embedding_dim: Dimension of learned embeddings
        """
        self.embedding_dim = embedding_dim
        self.embeddings_ = None
        self.vocab_size_ = None
        self.token_to_idx_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'FeatureEncoder':
        """
        Learn embeddings from data

        Args:
            X: Data to fit

        Returns:
            Self
        """
        unique_tokens = np.unique(X.flatten())
        self.vocab_size_ = len(unique_tokens)
        self.token_to_idx_ = {token: idx for idx, token in enumerate(unique_tokens)}

        self.embeddings_ = np.random.randn(
            self.vocab_size_, self.embedding_dim
        ) * 0.01

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using learned embeddings

        Args:
            X: Data to transform

        Returns:
            Embedded features
        """
        if not self.fitted:
            raise ValueError("FeatureEncoder must be fitted before transform")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        embedded = np.zeros((X.shape[0], X.shape[1], self.embedding_dim))

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                token = X[i, j]
                if token in self.token_to_idx_:
                    idx = self.token_to_idx_[token]
                    embedded[i, j] = self.embeddings_[idx]

        return embedded.reshape(X.shape[0], -1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform

        Returns:
            Embedded features
        """
        return self.fit(X).transform(X)

    def update_embeddings(self, embeddings: np.ndarray):
        """
        Update learned embeddings

        Args:
            embeddings: New embedding matrix
        """
        if not self.fitted:
            raise ValueError("FeatureEncoder must be fitted first")

        if embeddings.shape != self.embeddings_.shape:
            raise ValueError("Embedding shape mismatch")

        self.embeddings_ = embeddings


class TargetEncoder:
    """
    Encode categorical features using target statistics
    """

    def __init__(self, smoothing: float = 1.0):
        """
        Initialize target encoder

        Args:
            smoothing: Smoothing parameter
        """
        self.smoothing = smoothing
        self.encodings_ = None
        self.global_mean_ = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TargetEncoder':
        """
        Learn target encodings from data

        Args:
            X: Categorical features
            y: Target values

        Returns:
            Self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.global_mean_ = np.mean(y)
        self.encodings_ = []

        for i in range(X.shape[1]):
            category_stats = {}
            unique_categories = np.unique(X[:, i])

            for category in unique_categories:
                mask = X[:, i] == category
                category_targets = y[mask]

                n = len(category_targets)
                category_mean = np.mean(category_targets)

                smoothed_mean = (
                    (category_mean * n + self.global_mean_ * self.smoothing) /
                    (n + self.smoothing)
                )

                category_stats[category] = smoothed_mean

            self.encodings_.append(category_stats)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using learned target encodings

        Args:
            X: Data to transform

        Returns:
            Encoded features
        """
        if not self.fitted:
            raise ValueError("TargetEncoder must be fitted before transform")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded = np.zeros_like(X, dtype=float)

        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                category = X[j, i]
                if category in self.encodings_[i]:
                    encoded[j, i] = self.encodings_[i][category]
                else:
                    encoded[j, i] = self.global_mean_

        return encoded

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Data to fit and transform
            y: Target values

        Returns:
            Encoded features
        """
        return self.fit(X, y).transform(X)
