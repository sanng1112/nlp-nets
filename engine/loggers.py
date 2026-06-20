import os
import csv
from datetime import datetime
from typing import Optional


class CSVLogger:
    """
    Simple CSV logger for training metrics.

    Writes metrics (loss, accuracy, etc.) to a CSV file for later analysis
    or plotting. Each call to ``log_metrics`` appends a row.
    """

    def __init__(self, save_dir: str, filename: str = "training_log.csv"):
        """
        Args:
            save_dir: Directory to save the CSV file.
            filename: Name of the CSV file.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = os.path.join(save_dir, filename)
        self.file = open(self.filepath, "w", newline="")
        self.writer: Optional[csv.DictWriter] = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Write a row of metrics to the CSV file.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Optional training step number.
        """
        if step is not None:
            metrics["step"] = step
        metrics["timestamp"] = self.timestamp

        if self.writer is None:
            fieldnames = list(metrics.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
            self.writer.writeheader()

        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        """Close the CSV file."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
