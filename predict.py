"""
Prediction script for trained ML models.
"""
import torch
import numpy as np
from pathlib import Path
import argparse

from src.models.neural_network import FeedForwardNN
from src.inference.predictor import TorchPredictor, BatchPredictor
from src.preprocessing.scalers import FeatureScaler
from src.data.loader import DataSaver
from src.utils.helpers import load_pickle, get_device
from src.utils.logger import MLLogger


def main(
    checkpoint_path: str,
    input_data_path: str,
    output_path: str,
    model_config: dict = None,
    scaler_path: str = None
):
    """
    Main prediction function.

    Args:
        checkpoint_path: Path to model checkpoint
        input_data_path: Path to input data
        output_path: Path to save predictions
        model_config: Model configuration dictionary
        scaler_path: Path to saved scaler (optional)
    """
    logger = MLLogger('prediction')
    device = get_device()

    logger.info(f"Loading data from {input_data_path}")

    if input_data_path.endswith('.npy'):
        X = np.load(input_data_path)
    elif input_data_path.endswith('.npz'):
        data = np.load(input_data_path)
        X = data['features']
    else:
        import pandas as pd
        df = pd.read_csv(input_data_path)
        X = df.values

    logger.info(f"Data shape: {X.shape}")

    if scaler_path:
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = load_pickle(Path(scaler_path))
        X = scaler.transform(X)

    logger.info("Initializing model...")
    if model_config is None:
        model_config = {
            'input_dim': X.shape[1],
            'hidden_dims': [128, 64, 32],
            'output_dim': 1,
            'dropout_rate': 0.3,
            'activation': 'relu'
        }

    model = FeedForwardNN(**model_config)

    logger.info(f"Loading model from {checkpoint_path}")
    predictor = TorchPredictor.from_checkpoint(
        Path(checkpoint_path),
        model,
        device=str(device)
    )

    logger.info("Making predictions...")
    if len(X) > 10000:
        batch_predictor = BatchPredictor(model, device=str(device))
        predictions = batch_predictor.predict_large_dataset(X, show_progress=True)
    else:
        predictions = predictor.predict(X)

    logger.info(f"Predictions shape: {predictions.shape}")

    logger.info(f"Saving predictions to {output_path}")
    DataSaver.save_predictions(predictions, Path(output_path))

    logger.info("Prediction completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input data')
    parser.add_argument('--output', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--scaler', type=str, default=None, help='Path to saved scaler')
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
        input_data_path=args.input,
        output_path=args.output,
        model_config=model_config,
        scaler_path=args.scaler
    )
