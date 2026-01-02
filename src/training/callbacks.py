"""
Callback utilities for training
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


class Callback:
    """
    Base callback class
    """

    def on_train_begin(self):
        """Called at the beginning of training"""
        pass

    def on_train_end(self):
        """Called at the end of training"""
        pass

    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch"""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Called at the end of each epoch"""
        pass

    def on_batch_begin(self, batch: int):
        """Called at the beginning of each batch"""
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        """Called at the end of each batch"""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when metric stops improving
    """

    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 mode: str = 'min', min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping

        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_value = None
        self.best_weights = None
        self.wait = 0
        self.stop_training = False
        self.best_epoch = 0

    def on_train_begin(self):
        """Reset state at training start"""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Check if should stop training"""
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        if self.mode == 'min':
            is_improvement = current_value < (self.best_value - self.min_delta)
        else:
            is_improvement = current_value > (self.best_value + self.min_delta)

        if is_improvement:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                pass
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                print(f"\nEarly stopping triggered. Best epoch: {self.best_epoch}")


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints
    """

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True,
                 save_freq: int = 1):
        """
        Initialize model checkpoint

        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Whether to only save best model
            save_freq: Frequency of saving (in epochs)
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq

        self.best_value = None

    def on_train_begin(self):
        """Initialize best value"""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Save checkpoint if conditions met"""
        if (epoch + 1) % self.save_freq != 0:
            return

        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        should_save = False

        if self.save_best_only:
            if self.mode == 'min':
                if current_value < self.best_value:
                    self.best_value = current_value
                    should_save = True
            else:
                if current_value > self.best_value:
                    self.best_value = current_value
                    should_save = True
        else:
            should_save = True

        if should_save:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            print(f"\nSaving checkpoint to {filepath}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback
    """

    def __init__(self, schedule, verbose: bool = True):
        """
        Initialize LR scheduler

        Args:
            schedule: Function that takes epoch and returns learning rate
            verbose: Whether to print LR changes
        """
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int):
        """Update learning rate at epoch start"""
        new_lr = self.schedule(epoch)

        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate = {new_lr:.6f}")


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when metric stops improving
    """

    def __init__(self, monitor: str = 'val_loss', factor: float = 0.5,
                 patience: int = 5, mode: str = 'min',
                 min_lr: float = 1e-7, verbose: bool = True):
        """
        Initialize ReduceLROnPlateau

        Args:
            monitor: Metric to monitor
            factor: Factor to reduce LR by
            patience: Number of epochs to wait
            mode: 'min' or 'max'
            min_lr: Minimum learning rate
            verbose: Whether to print messages
        """
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_lr = min_lr
        self.verbose = verbose

        self.best_value = None
        self.wait = 0

    def on_train_begin(self):
        """Initialize best value"""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Check if should reduce learning rate"""
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        if self.mode == 'min':
            is_improvement = current_value < self.best_value
        else:
            is_improvement = current_value > self.best_value

        if is_improvement:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                if self.verbose:
                    print(f"\nReducing learning rate by factor {self.factor}")


class ProgressLogger(Callback):
    """
    Callback to log training progress
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize progress logger

        Args:
            log_file: Optional file to write logs
        """
        super().__init__()
        self.log_file = log_file
        self.logs = []

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Log epoch results"""
        log_entry = {'epoch': epoch + 1, **logs}
        self.logs.append(log_entry)

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'a') as f:
                f.write(str(log_entry) + '\n')
