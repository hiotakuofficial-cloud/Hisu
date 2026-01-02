"""Learning rate schedulers"""

import numpy as np


class LRScheduler:
    """Base learning rate scheduler"""

    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.epoch = 0

    def step(self) -> float:
        """Update learning rate"""
        raise NotImplementedError

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr


class StepLR(LRScheduler):
    """Step decay learning rate scheduler"""

    def __init__(self, initial_lr: float, step_size: int = 30, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> float:
        """Update learning rate every step_size epochs"""
        self.epoch += 1

        if self.epoch % self.step_size == 0:
            self.current_lr = self.current_lr * self.gamma

        return self.current_lr


class ExponentialLR(LRScheduler):
    """Exponential decay learning rate scheduler"""

    def __init__(self, initial_lr: float, gamma: float = 0.95):
        super().__init__(initial_lr)
        self.gamma = gamma

    def step(self) -> float:
        """Update learning rate exponentially"""
        self.epoch += 1
        self.current_lr = self.initial_lr * (self.gamma ** self.epoch)
        return self.current_lr


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler"""

    def __init__(self, initial_lr: float, T_max: int, eta_min: float = 0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self) -> float:
        """Update learning rate with cosine annealing"""
        self.epoch += 1
        self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                          (1 + np.cos(np.pi * self.epoch / self.T_max)) / 2
        return self.current_lr


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when metric plateaus"""

    def __init__(self, initial_lr: float, factor: float = 0.1, patience: int = 10, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = None
        self.num_bad_epochs = 0

    def step(self, metric: float) -> float:
        """Update learning rate based on metric"""
        self.epoch += 1

        if self.best_metric is None:
            self.best_metric = metric
            return self.current_lr

        if metric < self.best_metric:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self.num_bad_epochs = 0

        return self.current_lr


class WarmupScheduler(LRScheduler):
    """Warmup learning rate scheduler"""

    def __init__(self, initial_lr: float, warmup_epochs: int, target_lr: float):
        super().__init__(initial_lr)
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr

    def step(self) -> float:
        """Update learning rate with warmup"""
        self.epoch += 1

        if self.epoch < self.warmup_epochs:
            self.current_lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.epoch / self.warmup_epochs)
        else:
            self.current_lr = self.target_lr

        return self.current_lr


class CyclicLR(LRScheduler):
    """Cyclic learning rate scheduler"""

    def __init__(self, base_lr: float, max_lr: float, step_size: int = 2000):
        super().__init__(base_lr)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iteration = 0

    def step(self) -> float:
        """Update learning rate cyclically"""
        self.iteration += 1
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)
        self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        return self.current_lr
