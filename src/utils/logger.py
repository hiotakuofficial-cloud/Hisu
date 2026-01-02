"""
Logging utilities for ML experiments.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class MLLogger:
    """Custom logger for ML experiments."""

    def __init__(
        self,
        name: str = 'ml_pipeline',
        log_dir: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize ML logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Logging level for console
            file_level: Logging level for file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'{name}_{timestamp}.log'

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)


class ExperimentLogger:
    """Logger for tracking ML experiments."""

    def __init__(self, experiment_name: str, log_dir: Path):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_log = []
        self.config_log = {}

    def log_config(self, config: dict) -> None:
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.config_log = config
        config_file = self.log_dir / f'{self.experiment_name}_config.txt'

        with open(config_file, 'w') as f:
            for key, value in config.items():
                f.write(f'{key}: {value}\n')

    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        log_entry = {'epoch': epoch, **metrics}
        self.metrics_log.append(log_entry)

        metrics_file = self.log_dir / f'{self.experiment_name}_metrics.csv'

        if epoch == 0:
            with open(metrics_file, 'w') as f:
                headers = ','.join(log_entry.keys())
                f.write(headers + '\n')

        with open(metrics_file, 'a') as f:
            values = ','.join(str(v) for v in log_entry.values())
            f.write(values + '\n')

    def log_message(self, message: str) -> None:
        """
        Log a text message.

        Args:
            message: Message to log
        """
        log_file = self.log_dir / f'{self.experiment_name}_messages.txt'

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f'[{timestamp}] {message}\n')
