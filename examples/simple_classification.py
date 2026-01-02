"""
Simple classification example using neural networks
"""

import numpy as np
from src.models.neural_networks import FeedForwardNN
from src.data.dataset import CustomDataset, DataLoader
from src.training.trainer import SupervisedTrainer
from src.models.optimizers import Adam
from src.preprocessing.scalers import StandardScaler
from src.utils.helpers import set_seed


def main():
    """Simple classification example"""

    print("Simple Classification with Neural Network")
    print("-" * 50)

    set_seed(42)

    print("\nGenerating synthetic data...")
    X = np.random.randn(500, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_one_hot = np.eye(2)[y]

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y_one_hot.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(0.8 * len(X_scaled))
    X_train, y_train = X_scaled[:split_idx], y_one_hot[:split_idx]
    X_test, y_test = X_scaled[split_idx:], y_one_hot[split_idx:]

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("\nCreating model...")
    model = FeedForwardNN(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=2,
        activation='relu',
        learning_rate=0.01
    )

    optimizer = Adam(learning_rate=0.01)
    trainer = SupervisedTrainer(model, optimizer, loss_fn='cross_entropy')

    print("\nTraining model...")
    history = trainer.fit(train_loader, epochs=10, verbose=True)

    print("\nEvaluating model...")
    model.eval_mode()
    predictions = model.forward(X_test)
    pred_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(y_test, axis=-1)

    accuracy = np.mean(pred_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
