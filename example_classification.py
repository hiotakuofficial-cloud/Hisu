"""
Example: Classification workflow
"""

import numpy as np
from sklearn.datasets import make_classification
from src.data import DataLoader, DataPreprocessor
from src.models import Classifier
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator, ResultVisualizer
from src.utils import set_seed


def run_classification_example():
    """Run a complete classification workflow"""
    
    set_seed(42)
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
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
    
    classifier = Classifier(model_type='random_forest', n_estimators=100, random_state=42)
    classifier.build()
    
    trainer = ModelTrainer(classifier.model, task='classification')
    trainer.train(X_train_scaled, y_train)
    
    y_pred = classifier.predict(X_test_scaled)
    y_pred_proba = classifier.predict_proba(X_test_scaled)
    
    evaluator = ModelEvaluator(task='classification')
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba[:, 1])
    
    print("\nClassification Results:")
    evaluator.print_metrics()
    
    visualizer = ResultVisualizer()
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path='outputs/confusion_matrix.png'
    )
    
    classifier.save_model('models/classifier.pkl')
    
    print("\nClassification workflow completed successfully!")


if __name__ == "__main__":
    run_classification_example()
