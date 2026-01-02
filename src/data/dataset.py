"""
Dataset loading and management module
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import json
import pickle
from pathlib import Path


class CustomDataset:
    """
    Custom dataset class for ML training with neural networks
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize dataset

        Args:
            data: Input data array
            labels: Target labels array
            transform: Optional transformation function
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch of samples"""
        batch_data = self.data[indices]
        batch_labels = self.labels[indices]
        return batch_data, batch_labels


class DataLoader:
    """
    Data loader for batch processing with neural networks
    """

    def __init__(self, dataset: CustomDataset, batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False):
        """
        Initialize data loader

        Args:
            dataset: CustomDataset instance
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            if self.drop_last and end_idx - start_idx < self.batch_size:
                break

            batch_indices = indices[start_idx:end_idx]
            yield self.dataset.get_batch(batch_indices)


class DataManager:
    """
    Manager for data operations and persistence
    """

    @staticmethod
    def save_data(data: Dict[str, Any], filepath: str, format: str = 'pickle'):
        """Save data to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'npy':
            np.save(filepath, data)

    @staticmethod
    def load_data(filepath: str, format: str = 'pickle') -> Any:
        """Load data from file"""
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif format == 'npy':
            return np.load(filepath)

    @staticmethod
    def split_data(data: np.ndarray, labels: np.ndarray,
                   train_ratio: float = 0.8, val_ratio: float = 0.1,
                   shuffle: bool = True) -> Tuple:
        """
        Split data into train, validation, and test sets

        Args:
            data: Input data
            labels: Target labels
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            shuffle: Whether to shuffle before splitting

        Returns:
            Tuple of (train_data, train_labels, val_data, val_labels, test_data, test_labels)
        """
        num_samples = len(data)
        indices = np.arange(num_samples)

        if shuffle:
            np.random.shuffle(indices)

        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return (
            data[train_indices], labels[train_indices],
            data[val_indices], labels[val_indices],
            data[test_indices], labels[test_indices]
        )
