"""Cross-validation utilities"""

import numpy as np
from typing import List, Tuple, Optional


class KFold:
    """K-Fold cross-validation splitter"""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices"""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        splits = []

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            splits.append((train_indices, test_indices))
            current = stop

        return splits


class StratifiedKFold:
    """Stratified K-Fold cross-validation splitter"""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified train/test indices"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(X)
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]

        if self.shuffle:
            for indices in class_indices:
                np.random.shuffle(indices)

        class_counts = [len(indices) for indices in class_indices]
        class_fold_sizes = []

        for count in class_counts:
            fold_sizes = np.full(self.n_splits, count // self.n_splits, dtype=int)
            fold_sizes[:count % self.n_splits] += 1
            class_fold_sizes.append(fold_sizes)

        splits = []

        for fold_idx in range(self.n_splits):
            test_indices = []
            train_indices = []

            for class_idx, indices in enumerate(class_indices):
                fold_sizes = class_fold_sizes[class_idx]
                start = sum(fold_sizes[:fold_idx])
                stop = start + fold_sizes[fold_idx]

                test_indices.extend(indices[start:stop])
                train_indices.extend(np.concatenate([indices[:start], indices[stop:]]))

            test_indices = np.array(test_indices)
            train_indices = np.array(train_indices)

            splits.append((train_indices, test_indices))

        return splits


class CrossValidator:
    """Cross-validation executor"""

    def __init__(self, model, n_splits: int = 5, stratified: bool = False, shuffle: bool = True, random_state: Optional[int] = None):
        self.model = model
        self.n_splits = n_splits
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    def cross_validate(self, X: np.ndarray, y: np.ndarray, metric: str = 'accuracy') -> dict:
        """Perform cross-validation"""
        if self.stratified:
            splitter = StratifiedKFold(self.n_splits, self.shuffle, self.random_state)
        else:
            splitter = KFold(self.n_splits, self.shuffle, self.random_state)

        splits = splitter.split(X, y)

        scores = []
        fold_results = []

        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            if hasattr(self.model, 'fit'):
                self.model.fit(X_train, y_train)

            predictions = self.model.predict(X_test)

            score = self._compute_metric(y_test, predictions, metric)
            scores.append(score)

            fold_results.append({
                'fold': fold_idx + 1,
                'score': score,
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            })

        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_results': fold_results
        }

        return results

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Compute evaluation metric"""
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            pred_labels = np.argmax(y_pred, axis=1)
        else:
            pred_labels = (y_pred > 0.5).astype(int).flatten()

        if y_true.ndim > 1 and y_true.shape[1] > 1:
            true_labels = np.argmax(y_true, axis=1)
        else:
            true_labels = y_true.flatten()

        if metric == 'accuracy':
            return np.mean(pred_labels == true_labels)
        elif metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        else:
            return np.mean(pred_labels == true_labels)
