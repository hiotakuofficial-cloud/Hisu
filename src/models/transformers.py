"""Transformer architecture components"""

import numpy as np
from typing import Optional
from .neural_network import Layer, Dense


class Attention(Layer):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        batch_size = query.shape[0]

        Q = np.dot(query, self.W_q).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(key, self.W_k).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(value, self.W_v).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + (mask * -1e9)

        attention_weights = self._softmax(scores)

        context = np.matmul(attention_weights, V)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        self.output = np.dot(context, self.W_o)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Simplified backward pass"""
        return output_gradient

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class PositionalEncoding:
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model

        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding"""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class FeedForward(Layer):
    """Position-wise feed-forward network"""

    def __init__(self, d_model: int, d_ff: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input = input_data

        self.hidden = np.dot(input_data, self.W1) + self.b1
        self.hidden = np.maximum(0, self.hidden)

        self.output = np.dot(self.hidden, self.W2) + self.b2

        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass"""
        W2_gradient = np.dot(self.hidden.T, output_gradient)
        b2_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        hidden_gradient = np.dot(output_gradient, self.W2.T)
        hidden_gradient = hidden_gradient * (self.hidden > 0)

        W1_gradient = np.dot(self.input.T, hidden_gradient)
        b1_gradient = np.sum(hidden_gradient, axis=0, keepdims=True)

        self.W1 -= learning_rate * W1_gradient
        self.b1 -= learning_rate * b1_gradient
        self.W2 -= learning_rate * W2_gradient
        self.b2 -= learning_rate * b2_gradient

        return np.dot(hidden_gradient, self.W1.T)


class TransformerEncoderLayer:
    """Transformer encoder layer"""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048):
        self.attention = Attention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        attention_output = self.attention.forward(x, x, x, mask)
        x = self._layer_norm(x + attention_output, self.norm1_gamma, self.norm1_beta)

        ff_output = self.feed_forward.forward(x)
        x = self._layer_norm(x + ff_output, self.norm2_gamma, self.norm2_beta)

        return x

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta


class TransformerEncoder:
    """Transformer encoder"""

    def __init__(self, d_model: int, num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048, max_len: int = 5000):
        self.d_model = d_model
        self.num_layers = num_layers

        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer.forward(x, mask)

        return x


class TransformerDecoderLayer:
    """Transformer decoder layer"""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048):
        self.self_attention = Attention(d_model, num_heads)
        self.cross_attention = Attention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
        self.norm3_gamma = np.ones(d_model)
        self.norm3_beta = np.zeros(d_model)

    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                src_mask: Optional[np.ndarray] = None, tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        self_attention_output = self.self_attention.forward(x, x, x, tgt_mask)
        x = self._layer_norm(x + self_attention_output, self.norm1_gamma, self.norm1_beta)

        cross_attention_output = self.cross_attention.forward(x, encoder_output, encoder_output, src_mask)
        x = self._layer_norm(x + cross_attention_output, self.norm2_gamma, self.norm2_beta)

        ff_output = self.feed_forward.forward(x)
        x = self._layer_norm(x + ff_output, self.norm3_gamma, self.norm3_beta)

        return x

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta


class TransformerDecoder:
    """Transformer decoder"""

    def __init__(self, d_model: int, num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048, max_len: int = 5000):
        self.d_model = d_model
        self.num_layers = num_layers

        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [TransformerDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                src_mask: Optional[np.ndarray] = None, tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer.forward(x, encoder_output, src_mask, tgt_mask)

        return x
