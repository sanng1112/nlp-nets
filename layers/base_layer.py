import argparse
from typing import Any, Optional

from torch import nn


class BaseNLPLayer(nn.Module):
    """
    Base class for all NLP layers in nlp-nets.
    Provides common interfaces for argument parsing, weight initialization,
    and visualization hooks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def init_weights(self) -> None:
        """Initialize layer weights. Override in subclasses."""
        pass

    def extra_repr(self) -> str:
        return ""
