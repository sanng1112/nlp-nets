"""
Metrics builder — automatically constructs evaluation metrics based on task type.
"""

import torch
import torchmetrics
from torchmetrics import MetricCollection


def build_metrics(
    task_type: str,
    num_classes: int = None,
    ignore_index: int = -100,
) -> object:
    """
    Build evaluation metrics for a given task type.

    Args:
        task_type: Task type string (classification, mlm, causal_lm, seq2seq, etc.).
        num_classes: Number of classes (for classification tasks).
        ignore_index: Index to ignore in metrics computation.

    Returns:
        A torchmetrics MetricCollection or None if task type is unknown.
    """
    if not task_type:
        return None

    task_type = task_type.lower()

    if task_type == "classification":
        return MetricCollection({
            "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index),
            "F1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro", ignore_index=ignore_index),
            "Precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro", ignore_index=ignore_index),
            "Recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro", ignore_index=ignore_index),
        })
    elif task_type in ("mlm", "causal_lm", "seq2seq"):
        return MetricCollection({
            "Perplexity": torchmetrics.Perplexity(ignore_index=ignore_index),
            "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index) if num_classes else None,
        })
    elif task_type in ("seq_class", "sequence_classification"):
        return MetricCollection({
            "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index),
            "F1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro", ignore_index=ignore_index),
        })
    elif task_type == "regression":
        return MetricCollection({
            "MSE": torchmetrics.MeanSquaredError(),
            "MAE": torchmetrics.MeanAbsoluteError(),
        })
    else:
        return None
