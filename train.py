"""
Main training script for ML models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

from src.config.config import ExperimentConfig, get_default_config
from src.data.loader import DataLoader
from src.data.dataset import DataLoaderFactory, DataSplitter
from src.models.neural_network import FeedForwardNN
from src.training.trainer import Trainer
from src.training.optimizer import OptimizerFactory, SchedulerFactory
from src.preprocessing.scalers import FeatureScaler
from src.evaluation.metrics import ClassificationMetrics, RegressionMetrics
from src.utils.helpers import set_seed, get_device
from src.utils.logger import MLLogger, ExperimentLogger


def get_loss_function(loss_name: str) -> nn.Module:
    """Get loss function by name."""
    loss_functions = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'cross_entropy': nn.CrossEntropyLoss(),
        'bce': nn.BCEWithLogitsLoss(),
        'huber': nn.HuberLoss()
    }
    return loss_functions.get(loss_name, nn.MSELoss())


def main(config_path: str = None):
    """
    Main training function.

    Args:
        config_path: Path to configuration file (optional)
    """
    if config_path:
        config = ExperimentConfig.load(Path(config_path))
    else:
        config = get_default_config('classification')

    set_seed(config.seed)
    device = get_device()

    logger = MLLogger('training', log_dir=config.log_dir)
    exp_logger = ExperimentLogger(config.experiment_name, config.log_dir)

    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Using device: {device}")

    exp_logger.log_config(config.to_dict())

    logger.info("Loading and preprocessing data...")

    X = np.random.randn(1000, config.model.input_dim)
    y = np.random.randint(0, config.model.output_dim, size=1000)

    scaler = FeatureScaler(method=config.preprocessing.scaling_method)
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.split_data(
        X_scaled, y,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_state=config.seed
    )

    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    train_loader = DataLoaderFactory.create_loader(
        X_train, y_train,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=0
    )

    val_loader = DataLoaderFactory.create_loader(
        X_val, y_val,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info("Initializing model...")
    model = FeedForwardNN(
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        output_dim=config.model.output_dim,
        dropout_rate=config.model.dropout_rate,
        activation=config.model.activation
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = get_loss_function(config.training.loss_function)
    optimizer = OptimizerFactory.create_optimizer(
        model.parameters(),
        optimizer_name=config.training.optimizer,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    scheduler = None
    if config.training.scheduler:
        scheduler = SchedulerFactory.create_scheduler(
            optimizer,
            scheduler_name=config.training.scheduler,
            **config.training.scheduler_params
        )

    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=str(device),
        scheduler=scheduler
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        checkpoint_dir=config.checkpoint_dir if config.save_checkpoints else None
    )

    for epoch, (train_loss, val_loss) in enumerate(zip(history['train_losses'], history['val_losses'])):
        exp_logger.log_metrics(epoch, {'train_loss': train_loss, 'val_loss': val_loss})

    logger.info("Training completed!")

    logger.info("Evaluating on test set...")
    test_loader = DataLoaderFactory.create_loader(
        X_test, y_test,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loss = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")

    exp_logger.log_message(f"Final test loss: {test_loss:.4f}")

    logger.info(f"Experiment completed. Results saved to {config.log_dir}")


if __name__ == '__main__':
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
