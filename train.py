"""Training script for AI/ML models"""

import numpy as np
import argparse
from pathlib import Path

from src.models import NeuralNetwork, Dense, Dropout, CNN, RNN
from src.data import DataLoader
from src.preprocessing import StandardScaler, LabelEncoder
from src.training import Trainer, Adam, EarlyStopping, ModelCheckpoint
from src.evaluation import Evaluator
from src.utils import Logger, CheckpointManager
from config import ExperimentConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train AI/ML models')

    parser.add_argument('--model', type=str, default='neural_network',
                        choices=['neural_network', 'cnn', 'rnn'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)


def create_model(args):
    """Create model based on arguments"""
    if args.model == 'neural_network':
        model = NeuralNetwork()
        model.add(Dense(784, 256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, 128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, 10, activation='softmax'))
        model.compile(loss='cross_entropy', learning_rate=args.learning_rate)

    elif args.model == 'cnn':
        model = CNN(input_shape=(3, 224, 224), num_classes=10)
        model.compile(loss='cross_entropy', learning_rate=args.learning_rate)

    elif args.model == 'rnn':
        model = RNN(input_size=100, hidden_size=128, num_classes=10)
        model.compile(loss='cross_entropy', learning_rate=args.learning_rate)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def load_data(args):
    """Load and preprocess data"""
    np.random.seed(args.seed)

    X_train = np.random.randn(5000, 784)
    y_train = np.random.randint(0, 10, 5000)

    X_val = np.random.randn(1000, 784)
    y_val = np.random.randint(0, 10, 1000)

    X_test = np.random.randn(1000, 784)
    y_test = np.random.randint(0, 10, 1000)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    y_train_onehot = np.zeros((len(y_train), 10))
    y_train_onehot[np.arange(len(y_train)), y_train_encoded] = 1

    y_val_onehot = np.zeros((len(y_val), 10))
    y_val_onehot[np.arange(len(y_val)), y_val_encoded] = 1

    y_test_onehot = np.zeros((len(y_test), 10))
    y_test_onehot[np.arange(len(y_test)), y_test_encoded] = 1

    return (X_train, y_train_onehot), (X_val, y_val_onehot), (X_test, y_test_onehot)


def train_model(model, train_data, val_data, args, logger):
    """Train the model"""
    X_train, y_train = train_data
    X_val, y_val = val_data

    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }

    logger.log_training_start(config)

    trainer = Trainer(model, loss_fn='cross_entropy')

    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True
    )

    return history


def evaluate_model(model, test_data, logger):
    """Evaluate the model"""
    X_test, y_test = test_data

    evaluator = Evaluator(model)
    results = evaluator.evaluate(X_test, y_test, task_type='classification')

    logger.info("\nTest Set Results:")
    logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  F1-Score:  {results['f1_score']:.4f}")

    return results


def main():
    """Main training function"""
    args = parse_args()

    setup_directories(args)

    log_file = Path(args.log_dir) / 'training.log'
    logger = Logger('TrainingLogger', str(log_file))

    logger.info("Starting training pipeline...")

    np.random.seed(args.seed)

    logger.info("Loading data...")
    train_data, val_data, test_data = load_data(args)

    logger.info(f"Creating {args.model} model...")
    model = create_model(args)

    logger.info("Training model...")
    history = train_model(model, train_data, val_data, args, logger)

    logger.info("Evaluating model...")
    results = evaluate_model(model, test_data, logger)

    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    checkpoint_manager.save_checkpoint(
        state={'history': history, 'results': results},
        filename='final_model.pkl',
        metadata={'model': args.model, 'epochs': args.epochs}
    )

    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
