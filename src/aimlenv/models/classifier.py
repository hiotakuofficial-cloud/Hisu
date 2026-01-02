"""Neural network architectures for classification tasks."""

from typing import Literal

import torch
import torch.nn as nn


ModelName = Literal["mlp", "cnn"]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        return self.classifier(features)


def build_model(
    model_name: ModelName,
    dataset_name: Literal["mnist", "cifar10"],
    hidden_dim: int = 256,
    num_classes: int = 10,
) -> nn.Module:
    if model_name == "mlp":
        if dataset_name != "mnist":
            raise ValueError("MLP model is only supported for MNIST")
        return MLPClassifier(input_dim=28 * 28, hidden_dim=hidden_dim, num_classes=num_classes)
    if model_name == "cnn":
        input_channels = 1 if dataset_name == "mnist" else 3
        spatial_size = 28 if dataset_name == "mnist" else 32
        classifier = SimpleCNN(input_channels=input_channels, num_classes=num_classes)
        if dataset_name == "mnist":
            classifier.classifier[1] = nn.Linear(64 * 7 * 7, 128)
        else:
            classifier.classifier[1] = nn.Linear(64 * 8 * 8, 128)
        return classifier
    raise ValueError(f"Unsupported model: {model_name}")
