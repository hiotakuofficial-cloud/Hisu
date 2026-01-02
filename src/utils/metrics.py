"""Evaluation metrics"""

import numpy as np
from typing import Optional, List


class Metrics:
    """Collection of evaluation metrics"""

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        """Calculate precision"""
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            classes = np.unique(y_true)
            precisions = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            return np.mean(precisions)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        """Calculate recall"""
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            classes = np.unique(y_true)
            recalls = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            return np.mean(recalls)

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        """Calculate F1 score"""
        precision = Metrics.precision(y_true, y_pred, average)
        recall = Metrics.recall(y_true, y_pred, average)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for true, pred in zip(y_true, y_pred):
            true_idx = class_to_idx[true]
            pred_idx = class_to_idx[pred]
            cm[true_idx, pred_idx] += 1

        return cm

    @staticmethod
    def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate ROC AUC score"""
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]

        tps = np.cumsum(y_true)
        fps = np.cumsum(1 - y_true)

        tps = np.concatenate([[0], tps])
        fps = np.concatenate([[0], fps])

        if tps[-1] == 0 or fps[-1] == 0:
            return 0.5

        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        auc = np.trapz(tpr, fpr)

        return auc

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error"""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @staticmethod
    def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Log loss (cross-entropy loss)"""
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cohen's kappa score"""
        cm = Metrics.confusion_matrix(y_true, y_pred)
        n = np.sum(cm)
        po = np.trace(cm) / n
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n ** 2)
        return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cross-entropy loss"""
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            return -np.mean(y_true * np.log(y_pred))
        else:
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
