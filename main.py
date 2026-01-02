"""
Main entry point for AI/ML project
"""

import numpy as np
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer, FeatureSelector
from src.models import MLClassifier, MLRegressor, NeuralNetworkModel
from src.training import ModelTrainer, HyperparameterTuner
from src.evaluation import ModelEvaluator, ModelVisualizer
from src.utils import setup_logger, Config, set_random_seed


def main():
    """Main execution function"""
    logger = setup_logger('ml_project')
    logger.info("Starting AI/ML project")
    
    set_random_seed(42)
    
    logger.info("Project initialized successfully")
    logger.info("Available modules:")
    logger.info("  - Data: DataLoader, DataPreprocessor")
    logger.info("  - Features: FeatureEngineer, FeatureSelector")
    logger.info("  - Models: MLClassifier, MLRegressor, NeuralNetworkModel")
    logger.info("  - Training: ModelTrainer, HyperparameterTuner")
    logger.info("  - Evaluation: ModelEvaluator, ModelVisualizer")
    logger.info("  - Utils: Logger, Config, Helpers")


if __name__ == "__main__":
    main()
