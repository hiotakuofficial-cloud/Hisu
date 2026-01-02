"""Deep learning model architectures"""

import numpy as np
from typing import Tuple, Optional
from .neural_network import NeuralNetwork, Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout


class CNN(NeuralNetwork):
    """Convolutional Neural Network"""

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.add(Conv2D(input_channels=input_shape[0], output_channels=32, kernel_size=3, padding=1))
        self.add(MaxPooling2D(pool_size=2))
        self.add(Conv2D(input_channels=32, output_channels=64, kernel_size=3, padding=1))
        self.add(MaxPooling2D(pool_size=2))
        self.add(Flatten())

        flattened_size = 64 * (input_shape[1] // 4) * (input_shape[2] // 4)
        self.add(Dense(flattened_size, 128, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(128, num_classes, activation='softmax'))


class RNN(NeuralNetwork):
    """Recurrent Neural Network"""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        for _ in range(num_layers):
            self.add(LSTM(input_size if _ == 0 else hidden_size, hidden_size))
            self.add(Dropout(0.3))

        self.add(Dense(hidden_size, num_classes, activation='softmax'))


class Autoencoder:
    """Autoencoder for unsupervised learning"""

    def __init__(self, input_dim: int, encoding_dim: int):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        self.encoder = NeuralNetwork()
        self.encoder.add(Dense(input_dim, 128, activation='relu'))
        self.encoder.add(Dense(128, 64, activation='relu'))
        self.encoder.add(Dense(64, encoding_dim, activation='linear'))

        self.decoder = NeuralNetwork()
        self.decoder.add(Dense(encoding_dim, 64, activation='relu'))
        self.decoder.add(Dense(64, 128, activation='relu'))
        self.decoder.add(Dense(128, input_dim, activation='sigmoid'))

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input to latent space"""
        return self.encoder.forward(X, training=False)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode from latent space"""
        return self.decoder.forward(encoded, training=False)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through autoencoder"""
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return decoded

    def train_step(self, X: np.ndarray, learning_rate: float = 0.001) -> float:
        """Train autoencoder"""
        self.encoder.learning_rate = learning_rate
        self.decoder.learning_rate = learning_rate

        encoded = self.encoder.forward(X, training=True)
        decoded = self.decoder.forward(encoded, training=True)

        reconstruction_loss = np.mean((X - decoded) ** 2)

        loss_gradient = 2 * (decoded - X) / X.shape[0]
        self.decoder.backward(loss_gradient)

        encoder_gradient = self.decoder.layers[0].input
        self.encoder.backward(encoder_gradient)

        return reconstruction_loss


class GAN:
    """Generative Adversarial Network"""

    def __init__(self, latent_dim: int, output_dim: int):
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.generator = NeuralNetwork()
        self.generator.add(Dense(latent_dim, 128, activation='relu'))
        self.generator.add(Dense(128, 256, activation='relu'))
        self.generator.add(Dense(256, output_dim, activation='tanh'))

        self.discriminator = NeuralNetwork()
        self.discriminator.add(Dense(output_dim, 256, activation='relu'))
        self.discriminator.add(Dropout(0.3))
        self.discriminator.add(Dense(256, 128, activation='relu'))
        self.discriminator.add(Dropout(0.3))
        self.discriminator.add(Dense(128, 1, activation='sigmoid'))

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate samples"""
        noise = np.random.randn(n_samples, self.latent_dim)
        return self.generator.forward(noise, training=False)

    def train_step(self, real_data: np.ndarray, learning_rate: float = 0.0002) -> Tuple[float, float]:
        """Train GAN"""
        batch_size = real_data.shape[0]

        noise = np.random.randn(batch_size, self.latent_dim)
        fake_data = self.generator.forward(noise, training=True)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        self.discriminator.learning_rate = learning_rate
        d_loss_real = self.discriminator.train_step(real_data, real_labels)
        d_loss_fake = self.discriminator.train_step(fake_data, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2

        noise = np.random.randn(batch_size, self.latent_dim)
        misleading_labels = np.ones((batch_size, 1))

        self.generator.learning_rate = learning_rate
        fake_data = self.generator.forward(noise, training=True)
        discriminator_output = self.discriminator.forward(fake_data, training=False)

        g_loss = np.mean((discriminator_output - misleading_labels) ** 2)

        return d_loss, g_loss


class VAE:
    """Variational Autoencoder"""

    def __init__(self, input_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = NeuralNetwork()
        self.encoder.add(Dense(input_dim, 256, activation='relu'))
        self.encoder.add(Dense(256, 128, activation='relu'))

        self.mu_layer = Dense(128, latent_dim, activation='linear')
        self.logvar_layer = Dense(128, latent_dim, activation='linear')

        self.decoder = NeuralNetwork()
        self.decoder.add(Dense(latent_dim, 128, activation='relu'))
        self.decoder.add(Dense(128, 256, activation='relu'))
        self.decoder.add(Dense(256, input_dim, activation='sigmoid'))

    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to latent distribution parameters"""
        h = self.encoder.forward(X, training=False)
        mu = self.mu_layer.forward(h)
        logvar = self.logvar_layer.forward(h)
        return mu, logvar

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick"""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space"""
        return self.decoder.forward(z, training=False)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass"""
        mu, logvar = self.encode(X)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(self, X: np.ndarray, reconstruction: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> float:
        """Compute VAE loss"""
        reconstruction_loss = np.mean((X - reconstruction) ** 2)

        kl_divergence = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))
        kl_divergence /= X.shape[0]

        return reconstruction_loss + kl_divergence

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate samples"""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)
