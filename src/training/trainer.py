"""Model training orchestration"""

import numpy as np
from typing import Optional, List, Dict, Any
import time


class Trainer:
    """Training orchestrator for models"""

    def __init__(self, model, optimizer=None, loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model"""

        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            epoch_start = time.time()

            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            train_losses = []

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                if hasattr(self.model, 'train_step'):
                    loss = self.model.train_step(X_batch, y_batch)
                else:
                    predictions = self.model.forward(X_batch, training=True)
                    loss = self._compute_loss(predictions, y_batch)

                    loss_gradient = self._loss_gradient(predictions, y_batch)
                    self.model.backward(loss_gradient)

                train_losses.append(loss)

            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)

            train_predictions = self.model.predict(X_train)
            train_acc = self._compute_accuracy(train_predictions, y_train)
            self.history['train_acc'].append(train_acc)

            if X_val is not None and y_val is not None:
                val_predictions = self.model.predict(X_val)
                val_loss = self._compute_loss(val_predictions, y_val)
                val_acc = self._compute_accuracy(val_predictions, y_val)

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                if verbose:
                    epoch_time = time.time() - epoch_start
                    print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - "
                          f"loss: {avg_train_loss:.4f} - acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                if verbose:
                    epoch_time = time.time() - epoch_start
                    print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - "
                          f"loss: {avg_train_loss:.4f} - acc: {train_acc:.4f}")

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        predictions = self.model.predict(X_test)
        loss = self._compute_loss(predictions, y_test)
        acc = self._compute_accuracy(predictions, y_test)

        return {'loss': loss, 'accuracy': acc}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def _compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss"""
        if self.loss_fn == 'mse' or self.loss_fn is None:
            return np.mean((predictions - targets) ** 2)
        elif self.loss_fn == 'cross_entropy':
            predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
            if targets.ndim == 1 or targets.shape[1] == 1:
                return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
            else:
                return -np.mean(np.sum(targets * np.log(predictions), axis=1))
        elif self.loss_fn == 'mae':
            return np.mean(np.abs(predictions - targets))
        else:
            return np.mean((predictions - targets) ** 2)

    def _loss_gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute loss gradient"""
        if self.loss_fn == 'mse' or self.loss_fn is None:
            return 2 * (predictions - targets) / targets.shape[0]
        elif self.loss_fn == 'cross_entropy':
            return (predictions - targets) / targets.shape[0]
        elif self.loss_fn == 'mae':
            return np.sign(predictions - targets) / targets.shape[0]
        else:
            return 2 * (predictions - targets) / targets.shape[0]

    def _compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute accuracy"""
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            pred_labels = np.argmax(predictions, axis=1)
            if targets.ndim > 1 and targets.shape[1] > 1:
                true_labels = np.argmax(targets, axis=1)
            else:
                true_labels = targets
        else:
            pred_labels = (predictions > 0.5).astype(int).flatten()
            true_labels = targets.flatten()

        return np.mean(pred_labels == true_labels)
