"""Label and feature encoding"""

import numpy as np
from typing import Optional, List


class LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1"""

    def __init__(self):
        self.classes_ = None
        self.class_to_idx = None
        self.is_fitted = False

    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """Fit label encoder"""
        self.classes_ = np.unique(y)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        self.is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels to normalized encoding"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        return np.array([self.class_to_idx[label] for label in y])

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform labels"""
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels back to original encoding"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        return np.array([self.classes_[idx] for idx in y])


class OneHotEncoder:
    """Encode categorical features as one-hot numeric array"""

    def __init__(self):
        self.categories_ = None
        self.n_features_ = 0
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Fit one-hot encoder"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        self.n_features_ = X.shape[1]

        for i in range(self.n_features_):
            unique_values = np.unique(X[:, i])
            self.categories_.append(unique_values)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to one-hot encoding"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded = []

        for i in range(self.n_features_):
            n_categories = len(self.categories_[i])
            category_to_idx = {cat: idx for idx, cat in enumerate(self.categories_[i])}

            feature_encoded = np.zeros((len(X), n_categories))

            for j, value in enumerate(X[:, i]):
                if value in category_to_idx:
                    idx = category_to_idx[value]
                    feature_encoded[j, idx] = 1

            encoded.append(feature_encoded)

        return np.concatenate(encoded, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform one-hot encoding back to original"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        result = []
        start_idx = 0

        for i in range(self.n_features_):
            n_categories = len(self.categories_[i])
            end_idx = start_idx + n_categories

            one_hot_slice = X[:, start_idx:end_idx]
            indices = np.argmax(one_hot_slice, axis=1)
            decoded = np.array([self.categories_[i][idx] for idx in indices])

            result.append(decoded)
            start_idx = end_idx

        return np.column_stack(result) if len(result) > 1 else result[0]
