"""
Default configuration for ML experiments
"""

config = {
    'model': {
        'type': 'FeedForwardNN',
        'input_dim': 784,
        'hidden_dims': [256, 128, 64],
        'output_dim': 10,
        'activation': 'relu',
        'dropout': 0.2
    },

    'training': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'cross_entropy',
        'early_stopping': {
            'enabled': True,
            'patience': 10,
            'monitor': 'val_loss',
            'mode': 'min'
        },
        'checkpoint': {
            'enabled': True,
            'save_best_only': True,
            'monitor': 'val_loss',
            'mode': 'min'
        }
    },

    'data': {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'shuffle': True,
        'normalize': True,
        'augmentation': {
            'enabled': False,
            'noise_level': 0.01
        }
    },

    'preprocessing': {
        'scaler': 'standard',
        'feature_selection': {
            'enabled': False,
            'n_features': None
        },
        'pca': {
            'enabled': False,
            'n_components': None
        }
    },

    'logging': {
        'log_dir': 'logs',
        'experiment_name': 'default_experiment',
        'save_metrics': True,
        'verbose': True
    },

    'paths': {
        'data_dir': 'data',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'output_dir': 'outputs'
    },

    'random_seed': 42
}
