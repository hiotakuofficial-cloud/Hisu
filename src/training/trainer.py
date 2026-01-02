"""
Training pipeline for neural networks
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
import json
import time
from pathlib import Path

from .losses import LossFunctions
from .callbacks import Callback


class Trainer:
    """
    Base trainer class for neural network training
    """

    def __init__(self, model, optimizer, loss_fn: str = 'mse',
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize trainer

        Args:
            model: Neural network model
            optimizer: Optimizer instance
            loss_fn: Loss function name
            callbacks: List of callbacks
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = LossFunctions.get_loss(loss_fn)
        self.loss_grad = LossFunctions.get_loss_grad(loss_fn)
        self.callbacks = callbacks or []

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def train_step(self, batch_data: np.ndarray, batch_labels: np.ndarray) -> float:
        """
        Single training step

        Args:
            batch_data: Input batch
            batch_labels: Target batch

        Returns:
            Loss value
        """
        self.model.train_mode()

        predictions = self.model.forward(batch_data)
        loss = self.loss_fn(predictions, batch_labels)

        grad = self.loss_grad(predictions, batch_labels)
        self.model.backward(grad)
        self.model.update_weights()

        return loss

    def val_step(self, batch_data: np.ndarray, batch_labels: np.ndarray) -> float:
        """
        Single validation step

        Args:
            batch_data: Input batch
            batch_labels: Target batch

        Returns:
            Loss value
        """
        self.model.eval_mode()

        predictions = self.model.forward(batch_data)
        loss = self.loss_fn(predictions, batch_labels)

        return loss

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        epoch_loss = 0.0
        num_batches = 0

        for batch_data, batch_labels in train_loader:
            loss = self.train_step(batch_data, batch_labels)
            epoch_loss += loss
            num_batches += 1

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self, val_loader) -> float:
        """
        Validate for one epoch

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        epoch_loss = 0.0
        num_batches = 0

        for batch_data, batch_labels in val_loader:
            loss = self.val_step(batch_data, batch_labels)
            epoch_loss += loss
            num_batches += 1

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def fit(self, train_loader, val_loader=None, epochs: int = 10,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            verbose: Whether to print progress

        Returns:
            Training history
        """
        for callback in self.callbacks:
            callback.on_train_begin()

        for epoch in range(epochs):
            epoch_start = time.time()

            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)

            epoch_time = time.time() - epoch_start

            if verbose:
                log_str = f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - "
                log_str += f"train_loss: {train_loss:.4f}"
                if val_loss is not None:
                    log_str += f" - val_loss: {val_loss:.4f}"
                print(log_str)

            logs = {'train_loss': train_loss, 'val_loss': val_loss}

            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)

            should_stop = any(
                callback.stop_training
                for callback in self.callbacks
                if hasattr(callback, 'stop_training')
            )
            if should_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        for callback in self.callbacks:
            callback.on_train_end()

        return self.history

    def save_checkpoint(self, filepath: str, epoch: int, metadata: Optional[Dict] = None):
        """
        Save model checkpoint

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            metadata: Additional metadata
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_params': self.model.get_parameters(),
            'optimizer_state': self.optimizer.__dict__,
            'history': self.history,
            'metadata': metadata or {}
        }

        np.save(filepath, checkpoint)

    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = np.load(filepath, allow_pickle=True).item()

        self.model.set_parameters(checkpoint['model_params'])
        self.optimizer.__dict__.update(checkpoint['optimizer_state'])
        self.history = checkpoint['history']

        return checkpoint['epoch'], checkpoint['metadata']


class SupervisedTrainer(Trainer):
    """
    Trainer for supervised learning tasks
    """

    def __init__(self, model, optimizer, loss_fn: str = 'cross_entropy',
                 metric_fn: Optional[Callable] = None,
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize supervised trainer

        Args:
            model: Neural network model
            optimizer: Optimizer instance
            loss_fn: Loss function name
            metric_fn: Evaluation metric function
            callbacks: List of callbacks
        """
        super().__init__(model, optimizer, loss_fn, callbacks)
        self.metric_fn = metric_fn

    def compute_metrics(self, predictions: np.ndarray,
                       labels: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics

        Args:
            predictions: Model predictions
            labels: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        if self.metric_fn is not None:
            metrics['metric'] = self.metric_fn(predictions, labels)

        pred_classes = np.argmax(predictions, axis=-1)
        true_classes = np.argmax(labels, axis=-1) if labels.ndim > 1 else labels
        metrics['accuracy'] = np.mean(pred_classes == true_classes)

        return metrics


class UnsupervisedTrainer(Trainer):
    """
    Trainer for unsupervised learning tasks
    """

    def __init__(self, model, optimizer, loss_fn: str = 'mse',
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize unsupervised trainer

        Args:
            model: Neural network model
            optimizer: Optimizer instance
            loss_fn: Loss function name
            callbacks: List of callbacks
        """
        super().__init__(model, optimizer, loss_fn, callbacks)

    def train_step(self, batch_data: np.ndarray, batch_labels: np.ndarray = None) -> float:
        """
        Single training step for unsupervised learning

        Args:
            batch_data: Input batch
            batch_labels: Not used (for compatibility)

        Returns:
            Loss value
        """
        self.model.train_mode()

        reconstructed = self.model.forward(batch_data)
        loss = self.loss_fn(reconstructed, batch_data)

        grad = self.loss_grad(reconstructed, batch_data)
        self.model.backward(grad)
        self.model.update_weights()

        return loss


class ReinforcementTrainer(Trainer):
    """
    Trainer for reinforcement learning with neural networks
    """

    def __init__(self, model, optimizer, gamma: float = 0.99,
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize RL trainer

        Args:
            model: Neural network policy/value model
            optimizer: Optimizer instance
            gamma: Discount factor
            callbacks: List of callbacks
        """
        super().__init__(model, optimizer, 'mse', callbacks)
        self.gamma = gamma

    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Compute discounted returns

        Args:
            rewards: List of rewards

        Returns:
            Array of discounted returns
        """
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def train_episode(self, states: np.ndarray, actions: np.ndarray,
                     rewards: List[float]) -> float:
        """
        Train on single episode

        Args:
            states: Episode states
            actions: Episode actions
            rewards: Episode rewards

        Returns:
            Episode loss
        """
        self.model.train_mode()

        returns = self.compute_returns(rewards)

        action_probs = self.model.forward(states)
        selected_action_probs = action_probs[np.arange(len(actions)), actions]

        loss = -np.mean(np.log(selected_action_probs + 1e-8) * returns)

        grad = np.zeros_like(action_probs)
        grad[np.arange(len(actions)), actions] = -returns / (selected_action_probs + 1e-8)

        self.model.backward(grad)
        self.model.update_weights()

        return loss
