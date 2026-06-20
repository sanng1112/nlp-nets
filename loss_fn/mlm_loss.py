"""
Masked Language Modeling (MLM) loss function.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from loss_fn.base_criteria import BaseCriteria
from utils.registry import Registry


class MaskedLanguageModelingLoss(BaseCriteria):
    """
    MLM loss: CrossEntropy over masked positions only.

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
        loss = F.cross_entropy(
            prediction.view(-1, prediction.size(-1)),
            target.view(-1),
            ignore_index=-100,
        )
        return loss

    @classmethod
    def add_arguments(cls, parser: ...) -> ...:
        return parser


# Register
mlm_loss_registry = Registry("mlm", base_class=BaseCriteria)

@mlm_loss_registry.register(name="mlm_loss")
class MLMLossRegistered2(MaskedLanguageModelingLoss):
    """Registered MLM loss."""
    pass
