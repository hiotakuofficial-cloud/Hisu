"""
Activation functions for neural networks
"""

import numpy as np
from typing import Callable


class ActivationFunctions:
    """
    Collection of activation functions and their derivatives
    """

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        """ReLU gradient"""
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation"""
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU gradient"""
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_grad(x: np.ndarray) -> np.ndarray:
        """Sigmoid gradient"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        return np.tanh(x)

    @staticmethod
    def tanh_grad(x: np.ndarray) -> np.ndarray:
        """Tanh gradient"""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def softmax_grad(x: np.ndarray) -> np.ndarray:
        """Softmax gradient"""
        s = ActivationFunctions.softmax(x)
        return s * (1 - s)

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_grad(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU gradient"""
        return np.where(x > 0, 1, alpha * np.exp(x))

    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish activation"""
        return x * ActivationFunctions.sigmoid(x)

    @staticmethod
    def swish_grad(x: np.ndarray) -> np.ndarray:
        """Swish gradient"""
        s = ActivationFunctions.sigmoid(x)
        return s + x * s * (1 - s)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def gelu_grad(x: np.ndarray) -> np.ndarray:
        """GELU gradient approximation"""
        return ActivationFunctions.sigmoid(1.702 * x)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear activation (identity)"""
        return x

    @staticmethod
    def linear_grad(x: np.ndarray) -> np.ndarray:
        """Linear gradient"""
        return np.ones_like(x)

    @staticmethod
    def get_activation(name: str) -> Callable:
        """Get activation function by name"""
        activations = {
            'relu': ActivationFunctions.relu,
            'leaky_relu': ActivationFunctions.leaky_relu,
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'softmax': ActivationFunctions.softmax,
            'elu': ActivationFunctions.elu,
            'swish': ActivationFunctions.swish,
            'gelu': ActivationFunctions.gelu,
            'linear': ActivationFunctions.linear
        }
        return activations.get(name, ActivationFunctions.relu)

    @staticmethod
    def get_activation_grad(name: str) -> Callable:
        """Get activation gradient by name"""
        gradients = {
            'relu': ActivationFunctions.relu_grad,
            'leaky_relu': ActivationFunctions.leaky_relu_grad,
            'sigmoid': ActivationFunctions.sigmoid_grad,
            'tanh': ActivationFunctions.tanh_grad,
            'softmax': ActivationFunctions.softmax_grad,
            'elu': ActivationFunctions.elu_grad,
            'swish': ActivationFunctions.swish_grad,
            'gelu': ActivationFunctions.gelu_grad,
            'linear': ActivationFunctions.linear_grad
        }
        return gradients.get(name, ActivationFunctions.relu_grad)
