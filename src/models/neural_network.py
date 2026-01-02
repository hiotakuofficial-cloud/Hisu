"""Neural network layers and base model"""

import numpy as np
from typing import Optional, List, Tuple


class Layer:
    """Base layer class"""

    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer"""

    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input = input_data
        self.linear_output = np.dot(input_data, self.weights) + self.bias
        self.output = self._activate(self.linear_output)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        activation_gradient = self._activation_derivative(self.linear_output)
        delta = output_gradient * activation_gradient

        input_gradient = np.dot(delta, self.weights.T)
        weights_gradient = np.dot(self.input.T, delta)
        bias_gradient = np.sum(delta, axis=0, keepdims=True)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif self.activation == 'linear':
            return x
        else:
            return x

    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Activation function derivative"""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'softmax':
            return np.ones_like(x)
        elif self.activation == 'linear':
            return np.ones_like(x)
        else:
            return np.ones_like(x)


class Dropout(Layer):
    """Dropout layer for regularization"""

    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input = input_data

        if self.training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            self.output = input_data * self.mask
        else:
            self.output = input_data

        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        return output_gradient * self.mask if self.training else output_gradient


class Conv2D(Layer):
    """2D Convolutional layer"""

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros(output_channels)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input = input_data
        batch_size, channels, height, width = input_data.shape

        if self.padding > 0:
            input_data = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.output = np.zeros((batch_size, self.output_channels, out_height, out_width))

        for b in range(batch_size):
            for oc in range(self.output_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = input_data[b, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                        self.output[b, oc, h, w] = np.sum(receptive_field * self.kernels[oc]) + self.bias[oc]

        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        batch_size, _, height, width = self.input.shape
        input_gradient = np.zeros_like(self.input)

        kernels_gradient = np.zeros_like(self.kernels)
        bias_gradient = np.sum(output_gradient, axis=(0, 2, 3))

        input_data = self.input
        if self.padding > 0:
            input_data = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            input_gradient = np.pad(input_gradient, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        out_height, out_width = output_gradient.shape[2:]

        for b in range(batch_size):
            for oc in range(self.output_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = input_data[b, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]

                        kernels_gradient[oc] += output_gradient[b, oc, h, w] * receptive_field
                        input_gradient[b, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size] += output_gradient[b, oc, h, w] * self.kernels[oc]

        self.kernels -= learning_rate * kernels_gradient / batch_size
        self.bias -= learning_rate * bias_gradient / batch_size

        if self.padding > 0:
            input_gradient = input_gradient[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return input_gradient


class MaxPooling2D(Layer):
    """Max pooling layer"""

    def __init__(self, pool_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.trainable = False

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input = input_data
        batch_size, channels, height, width = input_data.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        pool_region = input_data[b, c, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]

                        self.output[b, c, h, w] = np.max(pool_region)
                        max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        self.max_indices[b, c, h, w] = [h_start + max_idx[0], w_start + max_idx[1]]

        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        input_gradient = np.zeros_like(self.input)
        batch_size, channels, out_height, out_width = output_gradient.shape

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_idx, w_idx = self.max_indices[b, c, h, w]
                        input_gradient[b, c, h_idx, w_idx] += output_gradient[b, c, h, w]

        return input_gradient


class Flatten(Layer):
    """Flatten layer"""

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input_shape = input_data.shape
        self.output = input_data.reshape(input_data.shape[0], -1)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        return output_gradient.reshape(self.input_shape)


class LSTM(Layer):
    """LSTM layer"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01

        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_length, _ = input_data.shape

        self.h = np.zeros((batch_size, seq_length + 1, self.hidden_size))
        self.c = np.zeros((batch_size, seq_length + 1, self.hidden_size))

        for t in range(seq_length):
            concat = np.concatenate([input_data[:, t, :], self.h[:, t, :]], axis=1)

            ft = self._sigmoid(np.dot(concat, self.Wf) + self.bf)
            it = self._sigmoid(np.dot(concat, self.Wi) + self.bi)
            c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)
            ot = self._sigmoid(np.dot(concat, self.Wo) + self.bo)

            self.c[:, t + 1, :] = ft * self.c[:, t, :] + it * c_tilde
            self.h[:, t + 1, :] = ot * np.tanh(self.c[:, t + 1, :])

        self.output = self.h[:, 1:, :]
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Simplified backward pass"""
        return output_gradient

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class GRU(Layer):
    """GRU layer"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * 0.01

        self.bz = np.zeros((1, hidden_size))
        self.br = np.zeros((1, hidden_size))
        self.bh = np.zeros((1, hidden_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_length, _ = input_data.shape

        self.h = np.zeros((batch_size, seq_length + 1, self.hidden_size))

        for t in range(seq_length):
            concat = np.concatenate([input_data[:, t, :], self.h[:, t, :]], axis=1)

            zt = self._sigmoid(np.dot(concat, self.Wz) + self.bz)
            rt = self._sigmoid(np.dot(concat, self.Wr) + self.br)

            concat_r = np.concatenate([input_data[:, t, :], rt * self.h[:, t, :]], axis=1)
            h_tilde = np.tanh(np.dot(concat_r, self.Wh) + self.bh)

            self.h[:, t + 1, :] = (1 - zt) * self.h[:, t, :] + zt * h_tilde

        self.output = self.h[:, 1:, :]
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Simplified backward pass"""
        return output_gradient

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class Embedding(Layer):
    """Embedding layer"""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input = input_data
        self.output = self.embeddings[input_data]
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        embedding_gradient = np.zeros_like(self.embeddings)

        np.add.at(embedding_gradient, self.input.flatten(), output_gradient.reshape(-1, self.embedding_dim))

        self.embeddings -= learning_rate * embedding_gradient

        return np.zeros_like(self.input)


class NeuralNetwork:
    """Neural network model"""

    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_function = None
        self.learning_rate = 0.001

    def add(self, layer: Layer):
        """Add layer to network"""
        self.layers.append(layer)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through network"""
        output = X

        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training
            output = layer.forward(output)

        return output

    def backward(self, loss_gradient: np.ndarray):
        """Backward pass through network"""
        gradient = loss_gradient

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, self.learning_rate)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Single training step"""
        predictions = self.forward(X, training=True)
        loss = self._compute_loss(predictions, y)
        loss_gradient = self._loss_gradient(predictions, y)

        self.backward(loss_gradient)

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X, training=False)

    def _compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss"""
        if self.loss_function == 'mse':
            return np.mean((predictions - targets) ** 2)
        elif self.loss_function == 'cross_entropy':
            predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
            return -np.mean(targets * np.log(predictions))
        else:
            return np.mean((predictions - targets) ** 2)

    def _loss_gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute loss gradient"""
        if self.loss_function == 'mse':
            return 2 * (predictions - targets) / targets.shape[0]
        elif self.loss_function == 'cross_entropy':
            return (predictions - targets) / targets.shape[0]
        else:
            return 2 * (predictions - targets) / targets.shape[0]

    def compile(self, loss: str = 'mse', learning_rate: float = 0.001):
        """Compile model"""
        self.loss_function = loss
        self.learning_rate = learning_rate
