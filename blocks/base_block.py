"""
Base class for all transformer blocks in nlp-nets.

Provides a common interface for blocks that wrap attention and feed-forward
layers with residual connections and normalization.
"""

from typing import Any, Optional

from torch import Tensor, nn

from layers.base_layer import BaseNLPLayer


class BaseTransformerBlock(BaseNLPLayer):
    """
    Abstract base for transformer blocks.

    Subclasses must implement forward() and can optionally override
    init_weights().
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask.

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        raise NotImplementedError

    def init_weights(self) -> None:
        """Initialize all submodules. Override in subclasses."""
        for module in self.modules():
            if hasattr(module, "init_weights") and module is not self:
                module.init_weights()
