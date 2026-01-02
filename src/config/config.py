"""
Configuration management for ML pipelines.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: Path = Path('data')
    train_file: str = 'train.csv'
    test_file: str = 'test.csv'
    val_file: Optional[str] = None
    target_column: str = 'target'
    feature_columns: Optional[List[str]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = 'feedforward'
    input_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    output_dim: int = 1
    dropout_rate: float = 0.3
    activation: str = 'relu'
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    weight_decay: float = 1e-5
    scheduler: Optional[str] = 'step'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {'step_size': 10, 'gamma': 0.1})
    early_stopping_patience: Optional[int] = 15
    gradient_clip_value: Optional[float] = None
    loss_function: str = 'mse'


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    scaling_method: str = 'standard'
    handle_missing: str = 'mean'
    outlier_method: Optional[str] = None
    feature_selection: bool = False
    n_features_to_select: Optional[int] = None
    dimensionality_reduction: Optional[str] = None
    n_components: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = 'ml_experiment'
    seed: int = 42
    device: str = 'cuda'
    output_dir: Path = Path('outputs')
    checkpoint_dir: Path = Path('checkpoints')
    log_dir: Path = Path('logs')
    save_checkpoints: bool = True
    save_frequency: int = 10
    verbose: bool = True

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def __post_init__(self):
        """Create necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save(self, file_path: Path) -> None:
        """Save configuration to file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4, default=str)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get('preprocessing', {}))

        base_config = {k: v for k, v in config_dict.items()
                      if k not in ['data', 'model', 'training', 'preprocessing']}

        return cls(
            **base_config,
            data=data_config,
            model=model_config,
            training=training_config,
            preprocessing=preprocessing_config
        )

    @classmethod
    def load(cls, file_path: Path) -> 'ExperimentConfig':
        """Load configuration from file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config(experiment_type: str = 'classification') -> ExperimentConfig:
    """
    Get default configuration for experiment type.

    Args:
        experiment_type: Type of experiment ('classification' or 'regression')

    Returns:
        ExperimentConfig with appropriate defaults
    """
    config = ExperimentConfig()

    if experiment_type == 'classification':
        config.training.loss_function = 'cross_entropy'
        config.model.output_dim = 2
    elif experiment_type == 'regression':
        config.training.loss_function = 'mse'
        config.model.output_dim = 1

    return config
