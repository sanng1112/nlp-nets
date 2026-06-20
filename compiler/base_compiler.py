"""
Base class for all model compilers in nlp-nets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseCompiler(ABC):
    """
    Abstract base class for model compilers.

    Subclasses implement a specific compilation backend (TorchScript, ONNX,
    torch.compile) and must provide ``compile()`` and ``save()``.

    Args:
        model: The PyTorch model to compile.
        opts: Configuration dictionary (typically loaded from YAML).
    """

    def __init__(self, model: torch.nn.Module, opts: Dict[str, Any]) -> None:
        self.model = model
        self.opts = opts
        self.compiled_model: Optional[torch.nn.Module] = None

    @abstractmethod
    def compile(self, *args, **kwargs) -> None:
        """
        Compile the model using the backend-specific approach.

        Must set ``self.compiled_model`` to the resulting artifact.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the compiled model artifact to disk.

        Args:
            path: Output file path (extension determines format).
        """
        ...

    def validate(self) -> bool:
        """
        Optional validation: run a forward pass on the compiled model
        to ensure it produces the expected output shapes.

        Returns:
            True if validation passes (or is not implemented).
        """
        return True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={type(self.model).__name__}, "
            f"compiled={self.compiled_model is not None})"
        )
