"""Experiment configuration"""

from .model_config import ModelConfig, CNNConfig, RNNConfig, TransformerConfig
from .training_config import TrainingConfig, OptimizerConfig, DataConfig, AugmentationConfig


class ExperimentConfig:
    """Complete experiment configuration"""

    def __init__(self, experiment_name: str = 'default_experiment'):
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.description = ''

        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.optimizer_config = OptimizerConfig()
        self.data_config = DataConfig()
        self.augmentation_config = AugmentationConfig()

        self.checkpoint_dir = 'checkpoints/'
        self.log_dir = 'logs/'
        self.output_dir = 'outputs/'

        self.device = 'cpu'
        self.random_seed = 42
        self.debug_mode = False

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'description': self.description,
            'model': self._config_to_dict(self.model_config),
            'training': self._config_to_dict(self.training_config),
            'optimizer': self._config_to_dict(self.optimizer_config),
            'data': self._config_to_dict(self.data_config),
            'augmentation': self._config_to_dict(self.augmentation_config),
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'output_dir': self.output_dir,
            'device': self.device,
            'random_seed': self.random_seed,
            'debug_mode': self.debug_mode
        }

    def _config_to_dict(self, config_obj) -> dict:
        """Convert config object to dictionary"""
        return {k: v for k, v in config_obj.__dict__.items() if not k.startswith('_')}

    def update_from_dict(self, config_dict: dict):
        """Update config from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ImageClassificationConfig(ExperimentConfig):
    """Image classification experiment configuration"""

    def __init__(self, experiment_name: str = 'image_classification'):
        super().__init__(experiment_name)
        self.model_config = CNNConfig()
        self.training_config.epochs = 50
        self.training_config.batch_size = 64
        self.augmentation_config.enabled = True


class TextClassificationConfig(ExperimentConfig):
    """Text classification experiment configuration"""

    def __init__(self, experiment_name: str = 'text_classification'):
        super().__init__(experiment_name)
        self.model_config = RNNConfig()
        self.training_config.epochs = 30
        self.training_config.batch_size = 32
        self.augmentation_config.enabled = False


class TimeSeriesForecastingConfig(ExperimentConfig):
    """Time series forecasting experiment configuration"""

    def __init__(self, experiment_name: str = 'time_series_forecasting'):
        super().__init__(experiment_name)
        self.model_config = RNNConfig()
        self.model_config.cell_type = 'lstm'
        self.training_config.epochs = 100
        self.training_config.batch_size = 16
        self.augmentation_config.enabled = False


class LanguageModelingConfig(ExperimentConfig):
    """Language modeling experiment configuration"""

    def __init__(self, experiment_name: str = 'language_modeling'):
        super().__init__(experiment_name)
        self.model_config = TransformerConfig()
        self.training_config.epochs = 50
        self.training_config.batch_size = 32
        self.training_config.learning_rate = 0.0001
