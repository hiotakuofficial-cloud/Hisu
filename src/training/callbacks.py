"""Training callbacks"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Callback:
    """Base callback class"""

    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of an epoch"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Called at the end of an epoch"""
        pass

    def on_batch_begin(self, batch: int):
        """Called at the beginning of a batch"""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None):
        """Called at the end of a batch"""
        pass

    def on_train_begin(self):
        """Called at the beginning of training"""
        pass

    def on_train_end(self):
        """Called at the end of training"""
        pass


class ModelCheckpoint(Callback):
    """Save model checkpoints during training"""

    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Save model if it's the best so far"""
        if logs is None:
            return

        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        if self.save_best_only:
            if self.best_score is None:
                self.best_score = current_score
                self._save_checkpoint(epoch, logs)
            else:
                is_improvement = (self.mode == 'min' and current_score < self.best_score) or \
                                 (self.mode == 'max' and current_score > self.best_score)

                if is_improvement:
                    self.best_score = current_score
                    self._save_checkpoint(epoch, logs)
        else:
            self._save_checkpoint(epoch, logs)

    def _save_checkpoint(self, epoch: int, logs: Dict[str, float]):
        """Save checkpoint to file"""
        checkpoint = {
            'epoch': epoch,
            'logs': logs
        }

        filepath = self.filepath.format(epoch=epoch + 1, **logs)

        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)


class HistoryLogger(Callback):
    """Log training history"""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.history = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': []
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Log metrics at end of epoch"""
        if logs is None:
            return

        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))

    def on_train_end(self):
        """Save history to file"""
        if self.filepath:
            Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self.history, f, indent=2)


class ProgressBar(Callback):
    """Display training progress bar"""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_begin(self, epoch: int):
        """Update progress at beginning of epoch"""
        self.current_epoch = epoch

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Display progress and metrics"""
        progress = (epoch + 1) / self.total_epochs
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)

        log_str = f"[{bar}] {epoch + 1}/{self.total_epochs}"

        if logs:
            metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in logs.items()])
            log_str += f" - {metrics_str}"

        print(f"\r{log_str}", end='')

        if epoch + 1 == self.total_epochs:
            print()


class LearningRateLogger(Callback):
    """Log learning rate changes"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.learning_rates = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Log current learning rate"""
        if hasattr(self.optimizer, 'learning_rate'):
            lr = self.optimizer.learning_rate
        elif hasattr(self.optimizer, 'current_lr'):
            lr = self.optimizer.current_lr
        else:
            lr = None

        if lr is not None:
            self.learning_rates.append(lr)
