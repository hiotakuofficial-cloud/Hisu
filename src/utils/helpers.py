"""
General utility helper functions.
"""
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to save file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: Path) -> None:
    """
    Save object using pickle.

    Args:
        obj: Object to save
        file_path: Path to save file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Path) -> Any:
    """
    Load object from pickle file.

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_directories(directories: list) -> None:
    """
    Create multiple directories if they don't exist.

    Args:
        directories: List of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """
    Get current timestamp as string.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class EarlyStopping:
    """Early stopping utility to stop training when metric stops improving."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current metric score

        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
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
            return True

        return False


class Timer:
    """Simple timer utility for measuring execution time."""

    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """Start the timer."""
        import time
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        import time
        self.end_time = time.time()
        return self.end_time - self.start_time

    def elapsed(self) -> float:
        """
        Get elapsed time without stopping.

        Returns:
            Elapsed time in seconds
        """
        import time
        return time.time() - self.start_time
