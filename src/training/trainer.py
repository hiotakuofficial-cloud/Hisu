"""
Training utilities for neural network models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, List
import numpy as np
from pathlib import Path


class Trainer:
    """Neural network trainer with flexible configuration."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> tuple:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average loss and metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_predictions.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader
    ) -> tuple:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                all_predictions.append(output.cpu())
                all_targets.append(target.cpu())

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}', end='')

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f' - Val Loss: {val_loss:.4f}', end='')

                # Early stopping
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0

                        if checkpoint_dir is not None:
                            self.save_checkpoint(checkpoint_dir / 'best_model.pt')
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f'\nEarly stopping triggered after epoch {epoch+1}')
                            break

            print()

            if self.scheduler is not None:
                self.scheduler.step()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
