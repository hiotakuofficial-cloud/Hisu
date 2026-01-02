"""Training configuration"""


class TrainingConfig:
    """Training hyperparameters configuration"""

    def __init__(self):
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        self.optimizer = 'adam'
        self.loss_function = 'cross_entropy'
        self.validation_split = 0.2
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 0.001
        self.lr_scheduler = 'step'
        self.lr_scheduler_params = {
            'step_size': 30,
            'gamma': 0.1
        }
        self.checkpoint_frequency = 5
        self.verbose = True


class OptimizerConfig:
    """Optimizer configuration"""

    def __init__(self):
        self.optimizer_type = 'adam'
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.weight_decay = 0.0001
        self.momentum = 0.9


class DataConfig:
    """Data configuration"""

    def __init__(self):
        self.data_path = 'data/'
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15
        self.shuffle = True
        self.random_seed = 42
        self.num_workers = 4
        self.pin_memory = True


class AugmentationConfig:
    """Data augmentation configuration"""

    def __init__(self):
        self.enabled = True
        self.random_flip = True
        self.random_rotation = True
        self.rotation_range = 30
        self.random_crop = True
        self.crop_size = (224, 224)
        self.random_brightness = True
        self.brightness_range = 0.2
        self.random_contrast = True
        self.contrast_range = (0.8, 1.2)
        self.noise_level = 0.01
