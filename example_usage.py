"""
Example usage of the AI/ML framework
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer, FeatureSelector
from src.models import MLClassifier, MLRegressor
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator, ModelVisualizer
from src.utils import setup_logger, set_random_seed


def classification_example():
    """Example classification pipeline"""
    set_random_seed(42)
    logger = setup_logger('classification_example')
    
    logger.info("Generating synthetic classification data")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    logger.info("Splitting data")
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y, test_size=0.2)
    
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.scale_features(X_train, method='standard')
    X_test_scaled = preprocessor.scaler.transform(X_test)
    
    logger.info("Training Random Forest classifier")
    classifier = MLClassifier(model_type='random_forest', n_estimators=100)
    classifier.train(X_train_scaled, y_train)
    
    logger.info("Making predictions")
    y_pred = classifier.predict(X_test_scaled)
    y_pred_proba = classifier.predict_proba(X_test_scaled)
    
    logger.info("Evaluating model")
    evaluator = ModelEvaluator(task='classification')
    results = evaluator.evaluate_classification(y_test, y_pred, y_pred_proba)
    
    logger.info(f"Classification Results: {results}")
    
    logger.info("Creating visualizations")
    visualizer = ModelVisualizer()
    visualizer.plot_confusion_matrix(
        y_test, y_pred, 
        save_path='outputs/visualizations/confusion_matrix.png'
    )
    visualizer.plot_roc_curve(
        y_test, y_pred_proba,
        save_path='outputs/visualizations/roc_curve.png'
    )
    
    logger.info("Classification example completed")


def regression_example():
    """Example regression pipeline"""
    set_random_seed(42)
    logger = setup_logger('regression_example')
    
    logger.info("Generating synthetic regression data")
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    logger.info("Splitting data")
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y, test_size=0.2)
    
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.scale_features(X_train, method='standard')
    X_test_scaled = preprocessor.scaler.transform(X_test)
    
    logger.info("Training Random Forest regressor")
    regressor = MLRegressor(model_type='random_forest', n_estimators=100)
    regressor.train(X_train_scaled, y_train)
    
    logger.info("Making predictions")
    y_pred = regressor.predict(X_test_scaled)
    
    logger.info("Evaluating model")
    evaluator = ModelEvaluator(task='regression')
    results = evaluator.evaluate_regression(y_test, y_pred)
    
    logger.info(f"Regression Results: {results}")
    
    logger.info("Creating visualizations")
    visualizer = ModelVisualizer()
    visualizer.plot_predictions_vs_actual(
        y_test, y_pred,
        save_path='outputs/visualizations/predictions_vs_actual.png'
    )
    visualizer.plot_residuals(
        y_test, y_pred,
        save_path='outputs/visualizations/residuals.png'
    )
    
    logger.info("Regression example completed")


if __name__ == "__main__":
    print("Running Classification Example...")
    classification_example()
    
    print("\nRunning Regression Example...")
    regression_example()
