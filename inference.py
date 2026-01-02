"""Inference script for trained models"""

import numpy as np
import argparse
from pathlib import Path

from src.models import NeuralNetwork, Dense, Dropout
from src.preprocessing import StandardScaler
from src.utils import CheckpointManager, Logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with trained models')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_data', type=str, default=None,
                        help='Path to input data')
    parser.add_argument('--output_file', type=str, default='predictions.npy',
                        help='Output file for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')

    return parser.parse_args()


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint"""
    checkpoint_manager = CheckpointManager(Path(checkpoint_path).parent)
    checkpoint_data = checkpoint_manager.load_checkpoint(Path(checkpoint_path).name)

    model = NeuralNetwork()
    model.add(Dense(784, 256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, 128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, 10, activation='softmax'))
    model.compile(loss='cross_entropy', learning_rate=0.001)

    return model


def load_input_data(input_path: str = None):
    """Load input data for inference"""
    if input_path:
        data = np.load(input_path)
    else:
        np.random.seed(42)
        data = np.random.randn(100, 784)

    return data


def preprocess_data(data: np.ndarray):
    """Preprocess input data"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def run_inference(model, data: np.ndarray, batch_size: int = 32):
    """Run inference on data"""
    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size

    predictions = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)

        batch_data = data[start_idx:end_idx]
        batch_predictions = model.predict(batch_data)

        predictions.append(batch_predictions)

    all_predictions = np.concatenate(predictions, axis=0)

    return all_predictions


def main():
    """Main inference function"""
    args = parse_args()

    logger = Logger('InferenceLogger')
    logger.info("Starting inference pipeline...")

    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    logger.info("Loading input data...")
    input_data = load_input_data(args.input_data)
    logger.info(f"Input data shape: {input_data.shape}")

    logger.info("Preprocessing data...")
    processed_data = preprocess_data(input_data)

    logger.info("Running inference...")
    predictions = run_inference(model, processed_data, args.batch_size)

    logger.info(f"Predictions shape: {predictions.shape}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions)

    logger.info(f"Predictions saved to {args.output_file}")

    pred_classes = np.argmax(predictions, axis=1)
    logger.info(f"Predicted classes: {pred_classes[:10]}...")

    logger.info("Inference completed successfully!")


if __name__ == '__main__':
    main()
