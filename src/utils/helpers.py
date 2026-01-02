"""
Helper utilities
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, filepath: str):
    """Save object to pickle file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str):
    """Save object to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filepath: str) -> Any:
    """Load object from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_directory(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except:
        pass
