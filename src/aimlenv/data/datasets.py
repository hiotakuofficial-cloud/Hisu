"""Dataset loading utilities using torchtext and torchvision."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DatasetName = Literal["mnist", "cifar10"]


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    name: DatasetName = "mnist"
    data_dir: Path = Path("./data")
    batch_size: int = 64
    val_split: float = 0.1
    download: bool = True
    num_workers: int = 2
    seed: int = 42

    def transform(self) -> transforms.Compose:
        if self.name == "mnist":
            return transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        if self.name == "cifar10":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            )
        raise ValueError(f"Unsupported dataset: {self.name}")


def load_dataset(config: DatasetConfig) -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(config.seed)

    if config.name == "mnist":
        dataset = datasets.MNIST(
            root=config.data_dir,
            train=True,
            transform=config.transform(),
            download=config.download,
        )
    elif config.name == "cifar10":
        dataset = datasets.CIFAR10(
            root=config.data_dir,
            train=True,
            transform=config.transform(),
            download=config.download,
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.name}")

    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader
