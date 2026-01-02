"""
Helper utility functions
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
import random


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, filepath: str, metadata: Optional[Dict] = None):
    """
    Save model to file

    Args:
        model: Model instance
        filepath: Path to save model
        metadata: Optional metadata
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'parameters': model.get_parameters(),
        'architecture': {
            'type': type(model).__name__,
            'input_dim': getattr(model, 'input_dim', None),
            'output_dim': getattr(model, 'output_dim', None),
            'hidden_dims': getattr(model, 'hidden_dims', None)
        },
        'metadata': metadata or {}
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Dict[str, Any]:
    """
    Load model from file

    Args:
        filepath: Path to model file

    Returns:
        Dictionary with model data
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    print(f"Model loaded from {filepath}")
    return model_data


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file

    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file

    Args:
        filepath: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)

    print(f"Configuration loaded from {filepath}")
    return config


def count_parameters(model) -> int:
    """
    Count total number of parameters in model

    Args:
        model: Model instance

    Returns:
        Total parameter count
    """
    total_params = 0

    for param in model.get_parameters():
        total_params += param.size

    return total_params


def format_time(seconds: float) -> str:
    """
    Format time in seconds to readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device_info() -> Dict[str, Any]:
    """
    Get device information

    Returns:
        Dictionary with device info
    """
    return {
        'platform': 'numpy',
        'compute': 'cpu',
        'precision': 'float64'
    }


def normalize_data(data: np.ndarray, mean: Optional[np.ndarray] = None,
                  std: Optional[np.ndarray] = None) -> tuple:
    """
    Normalize data using mean and standard deviation

    Args:
        data: Input data
        mean: Mean values (computed if None)
        std: Standard deviation values (computed if None)

    Returns:
        Tuple of (normalized_data, mean, std)
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)

    normalized = (data - mean) / (std + 1e-8)

    return normalized, mean, std


def denormalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalize data using mean and standard deviation

    Args:
        data: Normalized data
        mean: Mean values
        std: Standard deviation values

    Returns:
        Denormalized data
    """
    return data * std + mean


def one_hot_encode(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode labels

    Args:
        labels: Integer labels
        num_classes: Number of classes

    Returns:
        One-hot encoded labels
    """
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1

    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1

    return one_hot


def train_test_split(data: np.ndarray, labels: np.ndarray,
                    test_size: float = 0.2, shuffle: bool = True,
                    random_state: Optional[int] = None) -> tuple:
    """
    Split data into train and test sets

    Args:
        data: Input data
        labels: Target labels
        test_size: Proportion for test set
        shuffle: Whether to shuffle data
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(data)
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return (
        data[train_indices],
        data[test_indices],
        labels[train_indices],
        labels[test_indices]
    )


def create_batches(data: np.ndarray, batch_size: int) -> list:
    """
    Create batches from data

    Args:
        data: Input data
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    num_samples = len(data)
    batches = []

    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)

    return batches
