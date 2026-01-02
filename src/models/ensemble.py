"""Ensemble learning models"""

import numpy as np
from typing import List, Optional


class DecisionTree:
    """Decision tree for ensemble methods"""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Train decision tree"""
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            self.tree = {'leaf': True, 'value': np.mean(y)}
            return

        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        if best_gain == 0:
            self.tree = {'leaf': True, 'value': np.mean(y)}
            return

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        left_tree = DecisionTree(self.max_depth, self.min_samples_split)
        right_tree = DecisionTree(self.max_depth, self.min_samples_split)

        left_tree.fit(X[left_indices], y[left_indices], depth + 1)
        right_tree.fit(X[right_indices], y[right_indices], depth + 1)

        self.tree = {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray) -> float:
        """Predict single sample"""
        node = self.tree

        while not node['leaf']:
            if x[node['feature']] <= node['threshold']:
                node = node['left'].tree
            else:
                node = node['right'].tree

        return node['value']

    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        """Find best feature and threshold to split on"""
        best_gain = 0
        best_feature = 0
        best_threshold = 0

        parent_variance = np.var(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_var = np.var(y[left_mask])
                right_var = np.var(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)

                weighted_var = (n_left / n_total) * left_var + (n_right / n_total) * right_var
                gain = parent_variance - weighted_var

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain


class RandomForest:
    """Random Forest ensemble model"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train random forest"""
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = DecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def feature_importance(self, n_features: int) -> np.ndarray:
        """Calculate feature importance"""
        importance = np.zeros(n_features)

        for tree in self.trees:
            tree_importance = self._tree_feature_importance(tree.tree, n_features)
            importance += tree_importance

        return importance / self.n_estimators

    def _tree_feature_importance(self, node: dict, n_features: int) -> np.ndarray:
        """Calculate feature importance for a tree"""
        importance = np.zeros(n_features)

        if node['leaf']:
            return importance

        importance[node['feature']] += 1
        importance += self._tree_feature_importance(node['left'].tree, n_features)
        importance += self._tree_feature_importance(node['right'].tree, n_features)

        return importance


class GradientBoosting:
    """Gradient Boosting ensemble model"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: List[DecisionTree] = []
        self.initial_prediction = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train gradient boosting model"""
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - predictions

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=2)
            tree.fit(X, residuals)

            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions

            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        predictions = np.full(len(X), self.initial_prediction)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions

    def staged_predict(self, X: np.ndarray, n_trees: Optional[int] = None) -> np.ndarray:
        """Make predictions using first n_trees"""
        if n_trees is None:
            n_trees = len(self.trees)

        predictions = np.full(len(X), self.initial_prediction)

        for tree in self.trees[:n_trees]:
            predictions += self.learning_rate * tree.predict(X)

        return predictions
