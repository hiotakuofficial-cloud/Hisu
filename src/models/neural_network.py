"""
Neural network model architectures using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FeedForwardNN(nn.Module):
    """Fully connected feedforward neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize feedforward neural network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout_rate: Dropout probability
            activation: Activation function name
        """
        super(FeedForwardNN, self).__init__()

        self.activation_name = activation
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation_name == 'relu':
            return F.relu(x)
        elif self.activation_name == 'tanh':
            return torch.tanh(x)
        elif self.activation_name == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_name == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = self.get_activation(x)
            x = dropout(x)

        x = self.output_layer(x)
        return x


class ConvolutionalNN(nn.Module):
    """Convolutional Neural Network for image/sequence data."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: List[int] = [32, 64, 128],
        fc_dims: List[int] = [512, 256]
    ):
        """
        Initialize CNN.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            conv_channels: List of convolutional layer channels
            fc_dims: List of fully connected layer dimensions
        """
        super(ConvolutionalNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        prev_channels = input_channels
        for channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )
            prev_channels = channels

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        # Note: actual input dim needs to be calculated based on image size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        prev_dim = conv_channels[-1]
        for fc_dim in fc_dims:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, fc_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                )
            )
            prev_dim = fc_dim

        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x = self.output_layer(x)
        return x


class RecurrentNN(nn.Module):
    """Recurrent Neural Network (LSTM) for sequence data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM network.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            bidirectional: Use bidirectional LSTM
            dropout: Dropout probability
        """
        super(RecurrentNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output


class AutoEncoder(nn.Module):
    """AutoEncoder for unsupervised learning and dimensionality reduction."""

    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int],
        latent_dim: int
    ):
        """
        Initialize AutoEncoder.

        Args:
            input_dim: Input feature dimension
            encoding_dims: List of encoder layer dimensions
            latent_dim: Latent space dimension
        """
        super(AutoEncoder, self).__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(encoding_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
