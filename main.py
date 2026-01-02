"""Main entry point for AI/ML environment"""

import numpy as np
from src.models import NeuralNetwork, Dense, Dropout, CNN, RNN
from src.data import DataLoader, Dataset
from src.preprocessing import StandardScaler, LabelEncoder
from src.training import Trainer, Adam, EarlyStopping
from src.evaluation import Evaluator, CrossValidator
from src.utils import Metrics, Logger
from config import ExperimentConfig


def example_classification():
    """Example: Train a neural network for classification"""
    print("\n" + "=" * 60)
    print("EXAMPLE: NEURAL NETWORK CLASSIFICATION")
    print("=" * 60 + "\n")

    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 3, (1000,))

    X_test = np.random.randn(200, 20)
    y_test = np.random.randint(0, 3, (200,))

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    y_train_onehot = np.zeros((len(y_train), 3))
    y_train_onehot[np.arange(len(y_train)), y_train_encoded] = 1

    y_test_encoded = encoder.transform(y_test)
    y_test_onehot = np.zeros((len(y_test), 3))
    y_test_onehot[np.arange(len(y_test)), y_test_encoded] = 1

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = NeuralNetwork()
    model.add(Dense(20, 64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, 32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, 3, activation='softmax'))
    model.compile(loss='cross_entropy', learning_rate=0.001)

    trainer = Trainer(model, loss_fn='cross_entropy')
    history = trainer.train(X_train_scaled, y_train_onehot,
                            X_test_scaled, y_test_onehot,
                            epochs=20, batch_size=32, verbose=True)

    evaluator = Evaluator(model)
    results = evaluator.evaluate(X_test_scaled, y_test_onehot, task_type='classification')

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("=" * 60 + "\n")


def example_regression():
    """Example: Train a neural network for regression"""
    print("\n" + "=" * 60)
    print("EXAMPLE: NEURAL NETWORK REGRESSION")
    print("=" * 60 + "\n")

    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = np.sum(X_train, axis=1, keepdims=True) + np.random.randn(1000, 1) * 0.1

    X_test = np.random.randn(200, 10)
    y_test = np.sum(X_test, axis=1, keepdims=True) + np.random.randn(200, 1) * 0.1

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = NeuralNetwork()
    model.add(Dense(10, 64, activation='relu'))
    model.add(Dense(64, 32, activation='relu'))
    model.add(Dense(32, 1, activation='linear'))
    model.compile(loss='mse', learning_rate=0.001)

    trainer = Trainer(model, loss_fn='mse')
    history = trainer.train(X_train_scaled, y_train,
                            X_test_scaled, y_test,
                            epochs=20, batch_size=32, verbose=True)

    evaluator = Evaluator(model)
    results = evaluator.evaluate(X_test_scaled, y_test, task_type='regression')

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"MSE:  {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE:  {results['mae']:.4f}")
    print(f"R2:   {results['r2_score']:.4f}")
    print("=" * 60 + "\n")


def example_ensemble():
    """Example: Train ensemble models"""
    print("\n" + "=" * 60)
    print("EXAMPLE: ENSEMBLE LEARNING")
    print("=" * 60 + "\n")

    from src.models import RandomForest, GradientBoosting

    np.random.seed(42)
    X_train = np.random.randn(500, 10)
    y_train = (np.sum(X_train, axis=1) > 0).astype(float)

    X_test = np.random.randn(100, 10)
    y_test = (np.sum(X_test, axis=1) > 0).astype(float)

    print("Training Random Forest...")
    rf_model = RandomForest(n_estimators=50, max_depth=5)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = np.mean((rf_predictions > 0.5).astype(float) == y_test)

    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    print("\nTraining Gradient Boosting...")
    gb_model = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    gb_accuracy = np.mean((gb_predictions > 0.5).astype(float) == y_test)

    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    print("=" * 60 + "\n")


def example_cross_validation():
    """Example: Cross-validation"""
    print("\n" + "=" * 60)
    print("EXAMPLE: CROSS-VALIDATION")
    print("=" * 60 + "\n")

    from src.models import RandomForest

    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = (np.sum(X, axis=1) > 0).astype(float)

    model = RandomForest(n_estimators=30, max_depth=5)

    cv = CrossValidator(model, n_splits=5, shuffle=True, random_state=42)
    results = cv.cross_validate(X, y, metric='accuracy')

    print(f"Cross-Validation Scores: {results['scores']}")
    print(f"Mean Score: {results['mean_score']:.4f}")
    print(f"Std Score:  {results['std_score']:.4f}")
    print("=" * 60 + "\n")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("AI/ML ENVIRONMENT")
    print("Complete Machine Learning Framework")
    print("=" * 60 + "\n")

    example_classification()
    example_regression()
    example_ensemble()
    example_cross_validation()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
