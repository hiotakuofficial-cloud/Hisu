"""Optimization algorithms"""

import numpy as np
from typing import Dict, Any


class Optimizer:
    """Base optimizer class"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters"""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters with momentum"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        params += self.velocity

        return params


class Adam(Optimizer):
    """Adam optimizer"""

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters with adaptive learning rate"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params


class RMSprop(Optimizer):
    """RMSprop optimizer"""

    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters with adaptive learning rate"""
        if self.cache is None:
            self.cache = np.zeros_like(params)

        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * (gradients ** 2)
        params -= self.learning_rate * gradients / (np.sqrt(self.cache) + self.epsilon)

        return params


class AdaGrad(Optimizer):
    """AdaGrad optimizer"""

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters with accumulated gradients"""
        if self.cache is None:
            self.cache = np.zeros_like(params)

        self.cache += gradients ** 2
        params -= self.learning_rate * gradients / (np.sqrt(self.cache) + self.epsilon)

        return params


class Nadam(Optimizer):
    """Nadam optimizer (Nesterov-accelerated Adam)"""

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters with Nesterov momentum and adaptive learning rate"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        m_bar = self.beta1 * m_hat + (1 - self.beta1) * gradients / (1 - self.beta1 ** self.t)

        params -= self.learning_rate * m_bar / (np.sqrt(v_hat) + self.epsilon)

        return params
