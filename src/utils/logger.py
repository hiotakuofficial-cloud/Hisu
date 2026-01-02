"""Logging utilities"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """Custom logger for ML training"""

    def __init__(self, name: str = 'ML_Logger', log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

    def log_training_start(self, config: dict):
        """Log training start with config"""
        self.info("=" * 60)
        self.info("Training Started")
        self.info("=" * 60)
        for key, value in config.items():
            self.info(f"{key}: {value}")
        self.info("=" * 60)

    def log_epoch(self, epoch: int, total_epochs: int, metrics: dict):
        """Log epoch metrics"""
        metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch}/{total_epochs} - {metrics_str}")

    def log_training_end(self, total_time: float, best_metrics: dict):
        """Log training end"""
        self.info("=" * 60)
        self.info(f"Training Completed in {total_time:.2f} seconds")
        self.info("Best Metrics:")
        for key, value in best_metrics.items():
            self.info(f"  {key}: {value:.4f}")
        self.info("=" * 60)
