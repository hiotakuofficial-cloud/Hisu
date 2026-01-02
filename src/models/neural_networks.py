"""
Neural Network Architecture Implementations
All models use learned parameters - NO rule-based logic
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .layers import DenseLayer, ConvLayer, RecurrentLayer, AttentionLayer
from .activations import ActivationFunctions


class NeuralNetwork:
    """
    Base Neural Network class - pure learned model
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize neural network

        Args:
            learning_rate: Learning rate for optimization
        """
        self.layers = []
        self.learning_rate = learning_rate
        self.training = True

    def add_layer(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through network - all computations learned
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output, training=self.training)
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for gradient computation
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self):
        """Update all layer weights"""
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.update_weights(self.learning_rate)

    def train_mode(self):
        """Set network to training mode"""
        self.training = True

    def eval_mode(self):
        """Set network to evaluation mode"""
        self.training = False

    def get_parameters(self) -> List[np.ndarray]:
        """Get all network parameters"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append(layer.weights)
            if hasattr(layer, 'bias'):
                params.append(layer.bias)
        return params

    def set_parameters(self, params: List[np.ndarray]):
        """Set network parameters"""
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = params[param_idx]
                param_idx += 1
            if hasattr(layer, 'bias'):
                layer.bias = params[param_idx]
                param_idx += 1


class FeedForwardNN(NeuralNetwork):
    """
    Feedforward Neural Network - learns patterns from data
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, activation: str = 'relu',
                 learning_rate: float = 0.001):
        """
        Initialize feedforward network

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.add_layer(DenseLayer(prev_dim, hidden_dim, activation))
            prev_dim = hidden_dim

        self.add_layer(DenseLayer(prev_dim, output_dim, activation='linear'))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using learned parameters"""
        self.eval_mode()
        return self.forward(x)


class ConvolutionalNN(NeuralNetwork):
    """
    Convolutional Neural Network for spatial data
    All features learned through backpropagation
    """

    def __init__(self, input_shape: Tuple[int, ...],
                 conv_layers: List[Dict[str, Any]],
                 fc_layers: List[int],
                 output_dim: int,
                 learning_rate: float = 0.001):
        """
        Initialize CNN

        Args:
            input_shape: Input data shape
            conv_layers: List of convolutional layer configs
            fc_layers: Fully connected layer dimensions
            output_dim: Output dimension
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)

        self.input_shape = input_shape

        for conv_config in conv_layers:
            self.add_layer(ConvLayer(
                in_channels=conv_config['in_channels'],
                out_channels=conv_config['out_channels'],
                kernel_size=conv_config['kernel_size'],
                stride=conv_config.get('stride', 1),
                padding=conv_config.get('padding', 0),
                activation=conv_config.get('activation', 'relu')
            ))

        prev_dim = fc_layers[0] if fc_layers else output_dim
        for fc_dim in fc_layers[1:]:
            self.add_layer(DenseLayer(prev_dim, fc_dim, activation='relu'))
            prev_dim = fc_dim

        self.add_layer(DenseLayer(prev_dim, output_dim, activation='linear'))

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract learned features from input"""
        self.eval_mode()
        features = x
        for layer in self.layers[:-1]:
            features = layer.forward(features, training=False)
        return features


class RecurrentNN(NeuralNetwork):
    """
    Recurrent Neural Network for sequential data
    Uses learned temporal dependencies
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, num_layers: int = 1,
                 cell_type: str = 'lstm',
                 learning_rate: float = 0.001):
        """
        Initialize RNN

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension
            num_layers: Number of recurrent layers
            cell_type: Type of recurrent cell (lstm, gru, rnn)
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type

        prev_dim = input_dim
        for _ in range(num_layers):
            self.add_layer(RecurrentLayer(prev_dim, hidden_dim, cell_type))
            prev_dim = hidden_dim

        self.add_layer(DenseLayer(hidden_dim, output_dim, activation='linear'))

    def process_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Process sequential data using learned patterns"""
        self.eval_mode()
        return self.forward(sequence)


class TransformerModel(NeuralNetwork):
    """
    Transformer architecture with self-attention
    All attention patterns learned from data
    """

    def __init__(self, input_dim: int, d_model: int,
                 nhead: int, num_layers: int,
                 dim_feedforward: int, output_dim: int,
                 learning_rate: float = 0.001):
        """
        Initialize Transformer

        Args:
            input_dim: Input dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            output_dim: Output dimension
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead

        self.add_layer(DenseLayer(input_dim, d_model, activation='linear'))

        for _ in range(num_layers):
            self.add_layer(AttentionLayer(d_model, nhead))
            self.add_layer(DenseLayer(d_model, dim_feedforward, activation='relu'))
            self.add_layer(DenseLayer(dim_feedforward, d_model, activation='linear'))

        self.add_layer(DenseLayer(d_model, output_dim, activation='linear'))

    def attend(self, x: np.ndarray) -> np.ndarray:
        """Apply learned attention mechanism"""
        self.eval_mode()
        return self.forward(x)


class AutoEncoder(NeuralNetwork):
    """
    AutoEncoder for unsupervised learning
    Learns compressed representations
    """

    def __init__(self, input_dim: int, encoding_dims: List[int],
                 learning_rate: float = 0.001):
        """
        Initialize AutoEncoder

        Args:
            input_dim: Input dimension
            encoding_dims: Dimensions for encoding layers
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)

        prev_dim = input_dim
        for enc_dim in encoding_dims:
            self.add_layer(DenseLayer(prev_dim, enc_dim, activation='relu'))
            prev_dim = enc_dim

        decoding_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        for dec_dim in decoding_dims:
            self.add_layer(DenseLayer(prev_dim, dec_dim, activation='relu'))
            prev_dim = dec_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to learned latent space"""
        self.eval_mode()
        encoded = x
        for layer in self.layers[:len(self.layers)//2]:
            encoded = layer.forward(encoded, training=False)
        return encoded

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode from learned latent space"""
        self.eval_mode()
        decoded = encoded
        for layer in self.layers[len(self.layers)//2:]:
            decoded = layer.forward(decoded, training=False)
        return decoded
