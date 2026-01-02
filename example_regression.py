"""
Example: Regression workflow
"""

import numpy as np
from sklearn.datasets import make_regression
from src.data import DataLoader, DataPreprocessor
from src.models import Regressor
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator, ResultVisualizer
from src.utils import set_seed


def run_regression_example():
    """Run a complete regression workflow"""
    
    set_seed(42)
    
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        X, y, test_size=0.2, val_size=0.1
    )
    
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.scale_features(X_train, method='standard')
    X_val_scaled = preprocessor.scale_features(X_val, method='standard', fit=False)
    X_test_scaled = preprocessor.scale_features(X_test, method='standard', fit=False)
    
    regressor = Regressor(model_type='random_forest', n_estimators=100, random_state=42)
    regressor.build()
    
    trainer = ModelTrainer(regressor.model, task='regression')
    trainer.train(X_train_scaled, y_train)
    
    y_pred = regressor.predict(X_test_scaled)
    
    evaluator = ModelEvaluator(task='regression')
    metrics = evaluator.evaluate(y_test, y_pred)
    
    print("\nRegression Results:")
    evaluator.print_metrics()
    
    visualizer = ResultVisualizer()
    visualizer.plot_prediction_vs_actual(y_test, y_pred, save_path='outputs/predictions.png')
    visualizer.plot_residuals(y_test, y_pred, save_path='outputs/residuals.png')
    
    regressor.save_model('models/regressor.pkl')
    
    print("\nRegression workflow completed successfully!")


if __name__ == "__main__":
    run_regression_example()
