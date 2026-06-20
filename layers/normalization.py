from typing import Optional

import torch
from torch import nn, Tensor

from layers.base_layer import BaseNLPLayer


class LayerNorm(BaseNLPLayer):
    """
    Standard Layer Normalization.

    Args:
        normalized_shape: Input shape (int or tuple).
        eps: Small constant for numerical stability.
        elementwise_affine: Whether to learn affine parameters.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_norm(x)


class RMSLayerNorm(BaseNLPLayer):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Used in LLaMA and other modern transformers.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).to(x.dtype)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        if self.weight is not None:
            return f"hidden_size={self.weight.size(0)}, eps={self.eps}, elementwise_affine=True"
        return f"eps={self.eps}, elementwise_affine=False"
