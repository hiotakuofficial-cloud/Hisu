"""
Data augmentation techniques for neural network training
"""

import numpy as np
from typing import Callable, List, Optional, Tuple


class DataAugmentation:
    """
    Data augmentation for improving neural network generalization
    """

    def __init__(self, augmentation_funcs: Optional[List[Callable]] = None):
        """
        Initialize augmentation pipeline

        Args:
            augmentation_funcs: List of augmentation functions to apply
        """
        self.augmentation_funcs = augmentation_funcs or []

    def add_noise(self, data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to data"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def add_gaussian_noise(self, data: np.ndarray, mean: float = 0.0,
                          std: float = 0.1) -> np.ndarray:
        """Add Gaussian noise with specified parameters"""
        noise = np.random.normal(mean, std, data.shape)
        return data + noise

    def scale(self, data: np.ndarray, scale_range: Tuple = (0.8, 1.2)) -> np.ndarray:
        """Randomly scale data"""
        scale_factor = np.random.uniform(*scale_range)
        return data * scale_factor

    def rotate(self, data: np.ndarray, max_angle: float = 15) -> np.ndarray:
        """Apply rotation augmentation (for 2D data)"""
        if len(data.shape) < 2:
            return data

        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.deg2rad(angle)

        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        if data.shape[-1] == 2:
            return data @ rotation_matrix.T

        return data

    def flip(self, data: np.ndarray, axis: int = -1, probability: float = 0.5) -> np.ndarray:
        """Randomly flip data along axis"""
        if np.random.random() < probability:
            return np.flip(data, axis=axis)
        return data

    def mixup(self, data1: np.ndarray, data2: np.ndarray,
              label1: np.ndarray, label2: np.ndarray,
              alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: creates convex combinations of samples
        """
        lambda_param = np.random.beta(alpha, alpha)

        mixed_data = lambda_param * data1 + (1 - lambda_param) * data2
        mixed_label = lambda_param * label1 + (1 - lambda_param) * label2

        return mixed_data, mixed_label

    def cutout(self, data: np.ndarray, mask_size: int = 4) -> np.ndarray:
        """Apply cutout augmentation (masking random regions)"""
        augmented_data = data.copy()

        if len(data.shape) >= 2:
            h, w = data.shape[:2]
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - mask_size // 2, 0, h)
            y2 = np.clip(y + mask_size // 2, 0, h)
            x1 = np.clip(x - mask_size // 2, 0, w)
            x2 = np.clip(x + mask_size // 2, 0, w)

            augmented_data[y1:y2, x1:x2] = 0

        return augmented_data

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply all augmentation functions in sequence"""
        augmented_data = data.copy()

        for func in self.augmentation_funcs:
            augmented_data = func(augmented_data)

        return augmented_data

    def batch_augment(self, batch: np.ndarray) -> np.ndarray:
        """Apply augmentation to entire batch"""
        return np.array([self.apply(sample) for sample in batch])
