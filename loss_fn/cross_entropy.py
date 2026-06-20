"""
Cross-Entropy loss for NLP tasks (classification, causal LM).
"""

import argparse
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from loss_fn.base_criteria import BaseCriteria
from utils import logger
from utils.registry import Registry


class CrossEntropyLoss(BaseCriteria):
    """
    Cross-Entropy Loss with optional label smoothing and ignore_index.

    Args:
        opts: Configuration object.
        ignore_index: Index to ignore in the target (default: -100).
        label_smoothing: Label smoothing factor (default: 0.0).
    """

    def __init__(self, opts: Any, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        super().__init__(opts)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        input_sample: Any,
        prediction: Tensor,
        target: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        return F.cross_entropy(
            prediction.view(-1, prediction.size(-1)),
            target.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("CrossEntropyLoss")
        group.add_argument("--loss.cross-entropy.ignore-index", type=int, default=-100)
        group.add_argument("--loss.cross-entropy.label-smoothing", type=float, default=0.0)
        return parser


class MLMLoss(BaseCriteria):
    """
    Masked Language Modeling loss (CrossEntropy on masked positions only).

    Args:
        opts: Configuration object.
    """

    def __init__(self, opts: Any) -> None:
        super().__init__(opts)

    def forward(
        self,
        input_sample: Any,
        prediction: Tensor,
        target: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        # Only compute loss on masked positions (where target != -100)
        loss = F.cross_entropy(
            prediction.view(-1, prediction.size(-1)),
            target.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        return loss

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser


# Register losses
cls_loss_registry = Registry("classification", base_class=BaseCriteria)

@cls_loss_registry.register(name="cross_entropy")
class CrossEntropyLossRegistered(CrossEntropyLoss):
    """Registered version of CrossEntropyLoss."""
    pass


mlm_loss_registry = Registry("mlm", base_class=BaseCriteria)

@mlm_loss_registry.register(name="mlm_cross_entropy")
class MLMLossRegistered(MLMLoss):
    """Registered version of MLMLoss."""
    pass
