"""
Example usage of the AI/ML framework
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer, FeatureSelector
from src.models import MLClassifier, MLRegressor
from src.training import ModelTrainer, HyperparameterTuner
from src.evaluation import ModelEvaluator, ModelVisualizer
from src.utils import setup_logger, set_random_seed


def classification_example():
    """Example of classification workflow"""
    logger = setup_logger('classification_example')
    set_random_seed(42)
    
    logger.info("Generating synthetic classification data")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=0.2)
    
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.scale_features(X_train, method='standard')
    X_test_scaled = preprocessor.scale_features(X_test, method='standard')
    
    logger.info("Training Random Forest classifier")
    classifier = MLClassifier(model_type='random_forest', n_estimators=100)
    classifier.train(X_train_scaled, y_train)
    
    logger.info("Making predictions")
    y_pred = classifier.predict(X_test_scaled)
    y_pred_proba = classifier.predict_proba(X_test_scaled)
    
    logger.info("Evaluating model")
    evaluator = ModelEvaluator(task='classification')
    metrics = evaluator.evaluate_classification(y_test, y_pred, y_pred_proba)
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    visualizer = ModelVisualizer()
    visualizer.plot_confusion_matrix(y_test, y_pred, save_path='results/confusion_matrix.png')
    visualizer.plot_roc_curve(y_test, y_pred_proba, save_path='results/roc_curve.png')
    
    logger.info("Classification example completed")


def regression_example():
    """Example of regression workflow"""
    logger = setup_logger('regression_example')
    set_random_seed(42)
    
    logger.info("Generating synthetic regression data")
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=0.2)
    
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.scale_features(X_train, method='standard')
    X_test_scaled = preprocessor.scale_features(X_test, method='standard')
    
    logger.info("Training Random Forest regressor")
    regressor = MLRegressor(model_type='random_forest', n_estimators=100)
    regressor.train(X_train_scaled, y_train)
    
    logger.info("Making predictions")
    y_pred = regressor.predict(X_test_scaled)
    
    logger.info("Evaluating model")
    evaluator = ModelEvaluator(task='regression')
    metrics = evaluator.evaluate_regression(y_test, y_pred)
    
    logger.info(f"MSE: {metrics['mse']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"R2 Score: {metrics['r2_score']:.4f}")
    
    visualizer = ModelVisualizer()
    visualizer.plot_predictions_vs_actual(y_test, y_pred, save_path='results/predictions_vs_actual.png')
    visualizer.plot_residuals(y_test, y_pred, save_path='results/residuals.png')
    
    logger.info("Regression example completed")


def feature_engineering_example():
    """Example of feature engineering workflow"""
    logger = setup_logger('feature_engineering_example')
    set_random_seed(42)
    
    logger.info("Creating sample dataset")
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    X = data[['feature1', 'feature2', 'feature3']]
    y = data['target']
    
    logger.info("Creating interaction features")
    engineer = FeatureEngineer()
    X_with_interactions = engineer.create_interaction_features(X, ['feature1', 'feature2'])
    
    logger.info("Selecting best features")
    selector = FeatureSelector()
    X_selected = selector.select_k_best(X_with_interactions, y, k=5, task='classification')
    
    logger.info(f"Selected features: {selector.selected_features}")
    
    importance_df = selector.get_feature_importance(X_with_interactions, y, task='classification')
    logger.info(f"Feature importance:\n{importance_df.head()}")
    
    logger.info("Feature engineering example completed")


if __name__ == "__main__":
    print("Running Classification Example...")
    classification_example()
    
    print("\nRunning Regression Example...")
    regression_example()
    
    print("\nRunning Feature Engineering Example...")
    feature_engineering_example()
