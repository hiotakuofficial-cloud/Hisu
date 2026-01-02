"""Model configuration"""


class ModelConfig:
    """Base model configuration"""

    def __init__(self):
        self.model_type = 'neural_network'
        self.input_size = 784
        self.hidden_sizes = [128, 64]
        self.output_size = 10
        self.activation = 'relu'
        self.dropout_rate = 0.5


class CNNConfig:
    """CNN model configuration"""

    def __init__(self):
        self.model_type = 'cnn'
        self.input_shape = (3, 224, 224)
        self.num_classes = 10
        self.conv_layers = [
            {'filters': 32, 'kernel_size': 3, 'padding': 1},
            {'filters': 64, 'kernel_size': 3, 'padding': 1},
            {'filters': 128, 'kernel_size': 3, 'padding': 1}
        ]
        self.dense_layers = [256, 128]
        self.dropout_rate = 0.5


class RNNConfig:
    """RNN model configuration"""

    def __init__(self):
        self.model_type = 'rnn'
        self.input_size = 100
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = 10
        self.dropout_rate = 0.3
        self.cell_type = 'lstm'


class TransformerConfig:
    """Transformer model configuration"""

    def __init__(self):
        self.model_type = 'transformer'
        self.d_model = 512
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.d_ff = 2048
        self.dropout_rate = 0.1
        self.max_seq_length = 512


class EnsembleConfig:
    """Ensemble model configuration"""

    def __init__(self):
        self.model_type = 'ensemble'
        self.n_estimators = 100
        self.max_depth = 10
        self.min_samples_split = 2
        self.learning_rate = 0.1
        self.ensemble_method = 'random_forest'
