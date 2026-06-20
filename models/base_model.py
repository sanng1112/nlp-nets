import argparse
from typing import Any, Dict, Optional

import torch
from torch import nn


class BaseNLPModel(nn.Module):
    """
    Base class for all NLP models in nlp-nets.
    Provides common interface for forward pass, parameter counting,
    and device management.
    """

    def __init__(self, opts: Dict[str, Any], *args, **kwargs) -> None:
        super().__init__()
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self) -> Optional[nn.Module]:
        """Return the input embedding module (used for weight tying)."""
        return None

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embedding module (used for weight tying)."""
        pass

    def init_weights(self) -> None:
        """Initialize model weights. Override in subclasses."""
        pass
