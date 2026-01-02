"""
Logging utilities for ML experiments
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Logger:
    """
    Logger for ML experiments and training
    """

    def __init__(self, log_dir: str = 'logs', experiment_name: Optional[str] = None):
        """
        Initialize logger

        Args:
            log_dir: Directory for log files
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"

        self.metrics_history = []
        self.start_time = None

    def log(self, message: str, level: str = 'INFO'):
        """
        Log message to file

        Args:
            message: Message to log
            level: Log level
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        print(log_entry.strip())

    def log_metrics(self, metrics: Dict[str, Any], epoch: Optional[int] = None):
        """
        Log metrics to file

        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch number
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        if epoch is not None:
            entry['epoch'] = epoch

        self.metrics_history.append(entry)

        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters

        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        hp_file = self.log_dir / f"{self.experiment_name}_hyperparameters.json"

        with open(hp_file, 'w') as f:
            json.dump(hyperparameters, f, indent=2)

        self.log(f"Hyperparameters saved to {hp_file}")

    def start_timer(self):
        """Start experiment timer"""
        self.start_time = time.time()
        self.log("Experiment started")

    def end_timer(self):
        """End experiment timer and log duration"""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)

            self.log(f"Experiment completed in {hours}h {minutes}m {seconds}s")
            self.start_time = None

    def log_model_architecture(self, model_info: Dict[str, Any]):
        """
        Log model architecture

        Args:
            model_info: Dictionary with model information
        """
        arch_file = self.log_dir / f"{self.experiment_name}_architecture.json"

        with open(arch_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        self.log(f"Model architecture saved to {arch_file}")

    def get_metrics_history(self) -> list:
        """Get metrics history"""
        return self.metrics_history

    def save_summary(self, summary: Dict[str, Any]):
        """
        Save experiment summary

        Args:
            summary: Dictionary with experiment summary
        """
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log(f"Experiment summary saved to {summary_file}")
