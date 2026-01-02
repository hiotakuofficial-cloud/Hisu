"""
Optimization algorithms for neural network training
"""

import numpy as np
from typing import List, Dict, Any


class Optimizer:
    """
    Base optimizer class
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize optimizer

        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        """Update parameters"""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False):
        """
        Initialize SGD

        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        """Update parameters using SGD"""
        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.velocities:
                self.velocities[i] = np.zeros_like(param)

            v = self.velocities[i]
            v = self.momentum * v - self.learning_rate * grad

            if self.nesterov:
                param += self.momentum * v - self.learning_rate * grad
            else:
                param += v

            self.velocities[i] = v


class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        """Update parameters using Adam"""
        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """
    RMSprop optimizer with adaptive learning rates
    """

    def __init__(self, learning_rate: float = 0.001, alpha: float = 0.99,
                 epsilon: float = 1e-8):
        """
        Initialize RMSprop

        Args:
            learning_rate: Learning rate
            alpha: Smoothing constant
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.alpha = alpha
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        """Update parameters using RMSprop"""
        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.cache:
                self.cache[i] = np.zeros_like(param)

            self.cache[i] = self.alpha * self.cache[i] + (1 - self.alpha) * (grad ** 2)
            param -= self.learning_rate * grad / (np.sqrt(self.cache[i]) + self.epsilon)


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer with adaptive learning rates
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        """
        Initialize AdaGrad

        Args:
            learning_rate: Learning rate
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        """Update parameters using AdaGrad"""
        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.cache:
                self.cache[i] = np.zeros_like(param)

            self.cache[i] += grad ** 2
            param -= self.learning_rate * grad / (np.sqrt(self.cache[i]) + self.epsilon)


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.01):
        """
        Initialize AdamW

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay coefficient
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        """Update parameters using AdamW"""
        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * (
                m_hat / (np.sqrt(v_hat) + self.epsilon) +
                self.weight_decay * param
            )
