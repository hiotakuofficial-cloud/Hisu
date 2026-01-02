#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ai_ml_env.pipelines.training_pipeline import run_training
from ai_ml_env.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the training pipeline.')
    parser.add_argument('csv_path', type=str, help='Path to the training CSV data file.')
    parser.add_argument('target_column', type=str, help='Name of the target column in the dataset.')
    parser.add_argument('--model', type=str, default='logistic_regression', help='Model architecture to use.')
    parser.add_argument('--output', type=str, default=None, help='Optional custom path for saving the trained model.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    results = run_training(args.csv_path, args.target_column, model_name=args.model)

    model_path = Path(args.output) if args.output else results['model_path']
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(results['model_path']).replace(model_path)

    print('Training metrics:', results['metrics'])
    print('Model stored at:', model_path)


if __name__ == '__main__':
    main()
