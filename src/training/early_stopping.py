"""Early stopping to prevent overfitting"""

import numpy as np


class EarlyStopping:
    """Early stopping callback"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch: int, metric: float) -> bool:
        """Check if training should stop"""
        score = -metric if self.mode == 'min' else metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
