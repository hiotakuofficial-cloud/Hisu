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
    set_random_seed(42)
    logger = setup_logger('ml_project')
    
    logger.info("Starting AI/ML Pipeline")
    
    logger.info("Loading data...")
    data_loader = DataLoader()
    
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    logger.info("Engineering features...")
    feature_engineer = FeatureEngineer()
    
    logger.info("Selecting features...")
    feature_selector = FeatureSelector()
    
    logger.info("Training model...")
    
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(task='classification')
    
    logger.info("Visualizing results...")
    visualizer = ModelVisualizer()
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
