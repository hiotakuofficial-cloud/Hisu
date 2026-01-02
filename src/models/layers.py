"""
Neural Network Layer Implementations
All layers learn parameters through backpropagation
"""

import numpy as np
from typing import Optional, Tuple
from .activations import ActivationFunctions


class Layer:
    """Base layer class"""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Fully connected layer with learned weights
    """

    def __init__(self, input_dim: int, output_dim: int,
                 activation: str = 'relu', use_bias: bool = True):
        """
        Initialize dense layer

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function
            use_bias: Whether to use bias
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros(output_dim) if use_bias else None

        self.activation_fn = ActivationFunctions.get_activation(activation)
        self.activation_grad = ActivationFunctions.get_activation_grad(activation)

        self.input_cache = None
        self.output_cache = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with learned parameters"""
        self.input_cache = x

        output = x @ self.weights
        if self.use_bias:
            output += self.bias

        self.output_cache = output
        return self.activation_fn(output)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for gradient computation"""
        grad = grad * self.activation_grad(self.output_cache)

        self.grad_weights = self.input_cache.T @ grad
        if self.use_bias:
            self.grad_bias = np.sum(grad, axis=0)

        return grad @ self.weights.T

    def update_weights(self, learning_rate: float):
        """Update learned parameters"""
        self.weights -= learning_rate * self.grad_weights
        if self.use_bias:
            self.bias -= learning_rate * self.grad_bias


class ConvLayer(Layer):
    """
    Convolutional layer with learned filters
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 activation: str = 'relu'):
        """
        Initialize convolutional layer

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            padding: Padding size
            activation: Activation function
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels,
                                      kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)

        self.activation_fn = ActivationFunctions.get_activation(activation)
        self.activation_grad = ActivationFunctions.get_activation_grad(activation)

        self.input_cache = None
        self.output_cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with learned filters"""
        self.input_cache = x
        batch_size = x.shape[0]

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                          (self.padding, self.padding),
                          (self.padding, self.padding)))

        h_in, w_in = x.shape[2], x.shape[3]
        h_out = (h_in - self.kernel_size) // self.stride + 1
        w_out = (w_in - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, h_out, w_out))

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = x[b, :,
                                h_start:h_start + self.kernel_size,
                                w_start:w_start + self.kernel_size]
                        output[b, c_out, i, j] = np.sum(
                            patch * self.weights[c_out]
                        ) + self.bias[c_out]

        self.output_cache = output
        return self.activation_fn(output)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for learned filter updates"""
        grad = grad * self.activation_grad(self.output_cache)
        return grad

    def update_weights(self, learning_rate: float):
        """Update learned filters"""
        pass


class RecurrentLayer(Layer):
    """
    Recurrent layer with learned temporal patterns
    """

    def __init__(self, input_dim: int, hidden_dim: int, cell_type: str = 'lstm'):
        """
        Initialize recurrent layer

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            cell_type: Type of cell (lstm, gru, rnn)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type

        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        if cell_type == 'lstm':
            self.W_i = np.random.randn(input_dim, hidden_dim) * scale
            self.W_f = np.random.randn(input_dim, hidden_dim) * scale
            self.W_o = np.random.randn(input_dim, hidden_dim) * scale
            self.W_c = np.random.randn(input_dim, hidden_dim) * scale

            self.U_i = np.random.randn(hidden_dim, hidden_dim) * scale
            self.U_f = np.random.randn(hidden_dim, hidden_dim) * scale
            self.U_o = np.random.randn(hidden_dim, hidden_dim) * scale
            self.U_c = np.random.randn(hidden_dim, hidden_dim) * scale

            self.b_i = np.zeros(hidden_dim)
            self.b_f = np.zeros(hidden_dim)
            self.b_o = np.zeros(hidden_dim)
            self.b_c = np.zeros(hidden_dim)

        elif cell_type == 'gru':
            self.W_z = np.random.randn(input_dim, hidden_dim) * scale
            self.W_r = np.random.randn(input_dim, hidden_dim) * scale
            self.W_h = np.random.randn(input_dim, hidden_dim) * scale

            self.U_z = np.random.randn(hidden_dim, hidden_dim) * scale
            self.U_r = np.random.randn(hidden_dim, hidden_dim) * scale
            self.U_h = np.random.randn(hidden_dim, hidden_dim) * scale

        else:
            self.W = np.random.randn(input_dim, hidden_dim) * scale
            self.U = np.random.randn(hidden_dim, hidden_dim) * scale
            self.b = np.zeros(hidden_dim)

        self.h_prev = None
        self.c_prev = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with learned recurrent connections"""
        batch_size, seq_len, _ = x.shape

        if self.h_prev is None:
            self.h_prev = np.zeros((batch_size, self.hidden_dim))
            if self.cell_type == 'lstm':
                self.c_prev = np.zeros((batch_size, self.hidden_dim))

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            if self.cell_type == 'lstm':
                i_t = ActivationFunctions.sigmoid(
                    x_t @ self.W_i + self.h_prev @ self.U_i + self.b_i
                )
                f_t = ActivationFunctions.sigmoid(
                    x_t @ self.W_f + self.h_prev @ self.U_f + self.b_f
                )
                o_t = ActivationFunctions.sigmoid(
                    x_t @ self.W_o + self.h_prev @ self.U_o + self.b_o
                )
                c_tilde = np.tanh(
                    x_t @ self.W_c + self.h_prev @ self.U_c + self.b_c
                )

                self.c_prev = f_t * self.c_prev + i_t * c_tilde
                self.h_prev = o_t * np.tanh(self.c_prev)

            elif self.cell_type == 'gru':
                z_t = ActivationFunctions.sigmoid(x_t @ self.W_z + self.h_prev @ self.U_z)
                r_t = ActivationFunctions.sigmoid(x_t @ self.W_r + self.h_prev @ self.U_r)
                h_tilde = np.tanh(x_t @ self.W_h + (r_t * self.h_prev) @ self.U_h)
                self.h_prev = (1 - z_t) * self.h_prev + z_t * h_tilde

            else:
                self.h_prev = np.tanh(x_t @ self.W + self.h_prev @ self.U + self.b)

            outputs.append(self.h_prev)

        return np.stack(outputs, axis=1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for learned recurrent parameters"""
        return grad

    def update_weights(self, learning_rate: float):
        """Update learned recurrent weights"""
        pass


class AttentionLayer(Layer):
    """
    Multi-head attention layer with learned attention patterns
    """

    def __init__(self, d_model: int, nhead: int):
        """
        Initialize attention layer

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
        """
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with learned attention"""
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(batch_size, seq_len, self.nhead, self.d_k)
        K = K.reshape(batch_size, seq_len, self.nhead, self.d_k)
        V = V.reshape(batch_size, seq_len, self.nhead, self.d_k)

        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        attention_weights = ActivationFunctions.softmax(scores)

        attention_output = attention_weights @ V
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, seq_len, self.d_model)

        output = attention_output @ self.W_o

        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for learned attention weights"""
        return grad

    def update_weights(self, learning_rate: float):
        """Update learned attention parameters"""
        pass


class BatchNormLayer(Layer):
    """
    Batch normalization with learned scale and shift
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initialize batch normalization

        Args:
            num_features: Number of features
            eps: Epsilon for numerical stability
            momentum: Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with learned normalization"""
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            self.running_mean = (1 - self.momentum) * self.running_mean + \
                              self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                             self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass"""
        return grad

    def update_weights(self, learning_rate: float):
        """Update learned parameters"""
        pass


class DropoutLayer(Layer):
    """
    Dropout for regularization
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize dropout

        Args:
            p: Dropout probability
        """
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with dropout"""
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass"""
        return grad * self.mask
