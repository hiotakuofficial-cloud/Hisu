"""
Optimizer configurations and learning rate schedulers.
"""
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CyclicLR
)
from typing import Dict, Any


class OptimizerFactory:
    """Factory for creating optimizers with common configurations."""

    @staticmethod
    def create_optimizer(
        model_parameters,
        optimizer_name: str = 'adam',
        lr: float = 0.001,
        weight_decay: float = 0.0,
        **kwargs
    ) -> optim.Optimizer:
        """
        Create optimizer instance.

        Args:
            model_parameters: Model parameters to optimize
            optimizer_name: Name of optimizer
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Configured optimizer
        """
        optimizer_map = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adamax': optim.Adamax
        }

        if optimizer_name.lower() not in optimizer_map:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        optimizer_class = optimizer_map[optimizer_name.lower()]

        if optimizer_name.lower() == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            return optimizer_class(
                model_parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            return optimizer_class(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""

    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_name: str = 'step',
        **kwargs
    ):
        """
        Create learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            scheduler_name: Name of scheduler
            **kwargs: Scheduler-specific arguments

        Returns:
            Configured scheduler
        """
        if scheduler_name == 'step':
            step_size = kwargs.get('step_size', 10)
            gamma = kwargs.get('gamma', 0.1)
            return StepLR(optimizer, step_size=step_size, gamma=gamma)

        elif scheduler_name == 'exponential':
            gamma = kwargs.get('gamma', 0.95)
            return ExponentialLR(optimizer, gamma=gamma)

        elif scheduler_name == 'cosine':
            T_max = kwargs.get('T_max', 50)
            eta_min = kwargs.get('eta_min', 0)
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        elif scheduler_name == 'plateau':
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 10)
            return ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience
            )

        elif scheduler_name == 'cyclic':
            base_lr = kwargs.get('base_lr', 0.001)
            max_lr = kwargs.get('max_lr', 0.01)
            step_size_up = kwargs.get('step_size_up', 2000)
            return CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up
            )

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_optimizer_config(config_name: str = 'default') -> Dict[str, Any]:
    """
    Get predefined optimizer configuration.

    Args:
        config_name: Name of configuration preset

    Returns:
        Configuration dictionary
    """
    configs = {
        'default': {
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-5,
            'scheduler': 'step',
            'scheduler_params': {'step_size': 10, 'gamma': 0.1}
        },
        'aggressive': {
            'optimizer': 'adam',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'scheduler_params': {'T_max': 50}
        },
        'conservative': {
            'optimizer': 'sgd',
            'lr': 0.0001,
            'weight_decay': 1e-6,
            'momentum': 0.9,
            'scheduler': 'plateau',
            'scheduler_params': {'patience': 15, 'factor': 0.5}
        },
        'adaptive': {
            'optimizer': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.01,
            'scheduler': 'cyclic',
            'scheduler_params': {'base_lr': 0.0001, 'max_lr': 0.01}
        }
    }

    return configs.get(config_name, configs['default'])
