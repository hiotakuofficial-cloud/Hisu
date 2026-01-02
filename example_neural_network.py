"""
Example: Neural Network workflow
"""

import numpy as np
from sklearn.datasets import make_classification
from src.data import DataLoader, DataPreprocessor
from src.models import NeuralNetwork
from src.evaluation import ModelEvaluator, ResultVisualizer
from src.utils import set_seed


def run_neural_network_example():
    """Run a complete neural network workflow"""
    
    set_seed(42)
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
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
    
    nn_model = NeuralNetwork(
        input_size=20,
        hidden_layers=[64, 32, 16],
        output_size=1,
        task='binary_classification'
    )
    nn_model.build(dropout=0.3, activation='relu')
    
    nn_model.train(
        X_train_scaled,
        y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        verbose=True
    )
    
    y_pred_proba = nn_model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    evaluator = ModelEvaluator(task='classification')
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba.flatten())
    
    print("\nNeural Network Results:")
    evaluator.print_metrics()
    
    visualizer = ResultVisualizer()
    visualizer.plot_training_history(
        nn_model.training_history,
        save_path='outputs/training_history.png'
    )
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path='outputs/nn_confusion_matrix.png'
    )
    
    nn_model.save_model('models/neural_network.pth')
    
    print("\nNeural Network workflow completed successfully!")


if __name__ == "__main__":
    run_neural_network_example()
