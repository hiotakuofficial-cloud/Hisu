"""
Main entry point for AI/ML project
"""

import numpy as np
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer, FeatureSelector
from src.models import Classifier, Regressor, NeuralNetwork
from src.training import ModelTrainer, HyperparameterTuner
from src.evaluation import ModelEvaluator, ResultVisualizer
from src.utils import setup_logger, Config, set_seed


def main():
    """Main execution function"""
    
    set_seed(42)
    logger = setup_logger('ml_project', log_file='logs/project.log')
    logger.info("Starting AI/ML project")
    
    config = Config({
        'data': {
            'path': 'data/dataset.csv',
            'test_size': 0.2,
            'val_size': 0.1
        },
        'model': {
            'type': 'random_forest',
            'task': 'classification',
            'params': {
                'n_estimators': 100,
                'random_state': 42
            }
        },
        'training': {
            'cv_folds': 5,
            'scoring': 'accuracy'
        }
    })
    
    logger.info("Configuration loaded")
    
    logger.info("Project initialized successfully")
    logger.info("Ready for model training and evaluation")
    
    return config


if __name__ == "__main__":
    main()
