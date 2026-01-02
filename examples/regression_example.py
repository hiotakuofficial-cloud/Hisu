"""
Regression example using neural networks
"""

import numpy as np
from src.models.neural_networks import FeedForwardNN
from src.data.dataset import CustomDataset, DataLoader
from src.training.trainer import Trainer
from src.models.optimizers import Adam
from src.preprocessing.scalers import StandardScaler
from src.evaluation.metrics import RegressionMetrics
from src.utils.helpers import set_seed


def main():
    """Simple regression example"""

    print("Regression with Neural Network")
    print("-" * 50)

    set_seed(42)

    print("\nGenerating synthetic regression data...")
    X = np.random.randn(400, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(400) * 0.1
    y = y.reshape(-1, 1)

    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    split_idx = int(0.8 * len(X_scaled))
    X_train, y_train = X_scaled[:split_idx], y_scaled[:split_idx]
    X_test, y_test = X_scaled[split_idx:], y_scaled[split_idx:]

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("\nCreating regression model...")
    model = FeedForwardNN(
        input_dim=5,
        hidden_dims=[32, 16],
        output_dim=1,
        activation='relu',
        learning_rate=0.01
    )

    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model, optimizer, loss_fn='mse')

    print("\nTraining model...")
    history = trainer.fit(train_loader, epochs=15, verbose=True)

    print("\nEvaluating model...")
    model.eval_mode()
    predictions = model.forward(X_test)

    from src.evaluation.metrics import Metrics
    mse = Metrics.mean_squared_error(predictions, y_test)
    mae = Metrics.mean_absolute_error(predictions, y_test)
    r2 = Metrics.r2_score(predictions, y_test)

    print(f"\nTest Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")


if __name__ == "__main__":
    main()
