"""
Model evaluation script.
"""
import torch
import numpy as np
from pathlib import Path
import argparse

from src.models.neural_network import FeedForwardNN
from src.inference.predictor import TorchPredictor
from src.evaluation.metrics import ClassificationMetrics, RegressionMetrics
from src.preprocessing.scalers import FeatureScaler
from src.utils.helpers import load_pickle, get_device
from src.utils.logger import MLLogger
from src.utils.visualization import ModelVisualizer


def main(
    checkpoint_path: str,
    data_path: str,
    labels_path: str,
    task_type: str = 'classification',
    model_config: dict = None,
    scaler_path: str = None,
    output_dir: str = 'evaluation_results'
):
    """
    Main evaluation function.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to test data
        labels_path: Path to test labels
        task_type: 'classification' or 'regression'
        model_config: Model configuration dictionary
        scaler_path: Path to saved scaler
        output_dir: Directory to save evaluation results
    """
    logger = MLLogger('evaluation')
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    X = np.load(data_path) if data_path.endswith('.npy') else np.loadtxt(data_path, delimiter=',')
    y = np.load(labels_path) if labels_path.endswith('.npy') else np.loadtxt(labels_path, delimiter=',')

    logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    if scaler_path:
        logger.info("Applying scaling...")
        scaler = load_pickle(Path(scaler_path))
        X = scaler.transform(X)

    if model_config is None:
        model_config = {
            'input_dim': X.shape[1],
            'hidden_dims': [128, 64, 32],
            'output_dim': 1,
            'dropout_rate': 0.3,
            'activation': 'relu'
        }

    logger.info("Loading model...")
    model = FeedForwardNN(**model_config)
    predictor = TorchPredictor.from_checkpoint(
        Path(checkpoint_path),
        model,
        device=str(device)
    )

    logger.info("Making predictions...")
    predictions = predictor.predict(X)

    if task_type == 'classification':
        logger.info("Computing classification metrics...")

        if predictions.shape[1] > 1:
            y_pred = np.argmax(predictions, axis=1)
            y_proba = torch.softmax(torch.FloatTensor(predictions), dim=1).numpy()
        else:
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_proba = predictions.flatten()

        metrics = ClassificationMetrics.calculate_all_metrics(
            y, y_pred, y_proba if predictions.shape[1] > 1 else None
        )

        logger.info("Classification Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        cm = ClassificationMetrics.get_confusion_matrix(y, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")

        ModelVisualizer.plot_confusion_matrix(
            cm,
            title='Confusion Matrix',
            save_path=output_path / 'confusion_matrix.png'
        )

        report = ClassificationMetrics.get_classification_report(y, y_pred)
        logger.info(f"\nClassification Report:\n{report}")

        with open(output_path / 'metrics.txt', 'w') as f:
            f.write("Classification Metrics:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{cm}\n")
            f.write(f"\nClassification Report:\n{report}\n")

    else:
        logger.info("Computing regression metrics...")

        y_pred = predictions.flatten()
        metrics = RegressionMetrics.calculate_all_metrics(y, y_pred)

        logger.info("Regression Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        ModelVisualizer.plot_predictions_vs_actual(
            y, y_pred,
            title='Predictions vs Actual',
            save_path=output_path / 'predictions_vs_actual.png'
        )

        ModelVisualizer.plot_residuals(
            y, y_pred,
            title='Residual Plot',
            save_path=output_path / 'residuals.png'
        )

        with open(output_path / 'metrics.txt', 'w') as f:
            f.write("Regression Metrics:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

    logger.info(f"Evaluation completed! Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    parser.add_argument('--labels', type=str, required=True, help='Path to test labels')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'])
    parser.add_argument('--scaler', type=str, default=None, help='Path to saved scaler')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--input_dim', type=int, default=None, help='Model input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Model output dimension')

    args = parser.parse_args()

    model_config = None
    if args.input_dim:
        model_config = {
            'input_dim': args.input_dim,
            'hidden_dims': [128, 64, 32],
            'output_dim': args.output_dim,
            'dropout_rate': 0.3,
            'activation': 'relu'
        }

    main(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        labels_path=args.labels,
        task_type=args.task,
        model_config=model_config,
        scaler_path=args.scaler,
        output_dir=args.output
    )
