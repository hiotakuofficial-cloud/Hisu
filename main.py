"""
Main entry point for AI/ML environment
Example usage of the neural network framework
"""

import numpy as np
from src.models.neural_networks import FeedForwardNN
from src.data.dataset import DataManager, CustomDataset, DataLoader
from src.training.trainer import SupervisedTrainer
from src.models.optimizers import Adam
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import Logger
from src.utils.helpers import set_seed
from src.preprocessing.scalers import StandardScaler


def create_sample_data(num_samples=1000, input_dim=20, num_classes=3):
    """
    Create sample dataset for demonstration

    Args:
        num_samples: Number of samples
        input_dim: Input dimension
        num_classes: Number of classes

    Returns:
        Tuple of (data, labels)
    """
    np.random.seed(42)

    data = np.random.randn(num_samples, input_dim)

    labels = np.random.randint(0, num_classes, num_samples)
    labels_one_hot = np.zeros((num_samples, num_classes))
    labels_one_hot[np.arange(num_samples), labels] = 1

    return data, labels_one_hot


def main():
    """Main execution function"""

    print("=" * 70)
    print("AI/ML ENVIRONMENT - Neural Network Training Demo")
    print("=" * 70)

    set_seed(42)

    logger = Logger(log_dir='logs', experiment_name='demo_experiment')
    logger.start_timer()

    print("\n[1/6] Creating sample dataset...")
    X, y = create_sample_data(num_samples=1000, input_dim=20, num_classes=3)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    print("\n[2/6] Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_size = int(0.7 * len(X_scaled))
    val_size = int(0.15 * len(X_scaled))

    X_train, y_train = X_scaled[:train_size], y[:train_size]
    X_val, y_val = X_scaled[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X_scaled[train_size + val_size:], y[train_size + val_size:]

    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    print("\n[3/6] Creating data loaders...")
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("\n[4/6] Creating neural network model...")
    model = FeedForwardNN(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=3,
        activation='relu',
        learning_rate=0.001
    )

    print(f"Model architecture:")
    print(f"  Input dimension: 20")
    print(f"  Hidden layers: [64, 32]")
    print(f"  Output dimension: 3")
    print(f"  Activation: ReLU")

    print("\n[5/6] Training model...")
    optimizer = Adam(learning_rate=0.001)
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn='cross_entropy'
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        verbose=True
    )

    logger.log_metrics({
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1]
    })

    print("\n[6/6] Evaluating model...")
    evaluator = ModelEvaluator(model, task_type='classification')
    test_results = evaluator.evaluate(test_loader, verbose=True)

    logger.log_metrics(test_results, epoch='final')

    logger.end_timer()

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)

    print("\nModel Summary:")
    print(f"  Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Test Accuracy: {test_results.get('accuracy', 0):.4f}")

    return model, history, test_results


if __name__ == "__main__":
    model, history, results = main()
