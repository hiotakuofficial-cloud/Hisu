"""
Plotting utilities for ML models
"""

import numpy as np
from typing import Dict, List, Optional, Any


class Plotter:
    """
    Base plotter class for visualizations
    """

    def __init__(self, save_dir: str = 'plots'):
        """
        Initialize plotter

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir

    def plot_loss_curve(self, train_loss: List[float], val_loss: Optional[List[float]] = None):
        """
        Display loss curves

        Args:
            train_loss: Training loss history
            val_loss: Validation loss history
        """
        print("\n=== Loss Curves ===")
        print(f"Training Loss (last 10 epochs): {train_loss[-10:]}")
        if val_loss:
            print(f"Validation Loss (last 10 epochs): {val_loss[-10:]}")

        if train_loss:
            print(f"Final Training Loss: {train_loss[-1]:.6f}")
        if val_loss:
            print(f"Final Validation Loss: {val_loss[-1]:.6f}")

    def plot_metrics(self, metrics: Dict[str, List[float]]):
        """
        Display metrics over time

        Args:
            metrics: Dictionary of metric histories
        """
        print("\n=== Metrics History ===")
        for metric_name, values in metrics.items():
            if values:
                print(f"{metric_name}:")
                print(f"  Last 5 values: {values[-5:]}")
                print(f"  Best: {max(values):.6f}, Worst: {min(values):.6f}")


class TrainingPlotter:
    """
    Plotter for training progress
    """

    def __init__(self):
        """Initialize training plotter"""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def update(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
               train_acc: Optional[float] = None, val_acc: Optional[float] = None):
        """
        Update training history

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
        """
        self.history['train_loss'].append(train_loss)

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)

    def display_progress(self, epoch: int, total_epochs: int):
        """
        Display training progress

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
        """
        progress = (epoch + 1) / total_epochs * 100

        print(f"\nEpoch {epoch + 1}/{total_epochs} ({progress:.1f}% complete)")

        if self.history['train_loss']:
            print(f"Training Loss: {self.history['train_loss'][-1]:.6f}")
        if self.history['val_loss']:
            print(f"Validation Loss: {self.history['val_loss'][-1]:.6f}")
        if self.history['train_acc']:
            print(f"Training Accuracy: {self.history['train_acc'][-1]:.4f}")
        if self.history['val_acc']:
            print(f"Validation Accuracy: {self.history['val_acc'][-1]:.4f}")

    def plot_summary(self):
        """Display training summary"""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        if self.history['train_loss']:
            print(f"\nFinal Training Loss: {self.history['train_loss'][-1]:.6f}")
            print(f"Best Training Loss: {min(self.history['train_loss']):.6f}")

        if self.history['val_loss']:
            print(f"\nFinal Validation Loss: {self.history['val_loss'][-1]:.6f}")
            print(f"Best Validation Loss: {min(self.history['val_loss']):.6f}")

        if self.history['train_acc']:
            print(f"\nFinal Training Accuracy: {self.history['train_acc'][-1]:.4f}")
            print(f"Best Training Accuracy: {max(self.history['train_acc']):.4f}")

        if self.history['val_acc']:
            print(f"\nFinal Validation Accuracy: {self.history['val_acc'][-1]:.4f}")
            print(f"Best Validation Accuracy: {max(self.history['val_acc']):.4f}")

        print("=" * 60)
