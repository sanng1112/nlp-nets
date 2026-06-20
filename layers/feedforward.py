from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.base_layer import BaseNLPLayer


class PositionwiseFeedForward(BaseNLPLayer):
    """
    Standard position-wise feed-forward network (MLP with GeLU activation).

    Args:
        hidden_size: Model dimension.
        intermediate_size: Inner dimension of the FFN.
        dropout: Dropout probability after activation.
        activation: Activation function ('gelu' or 'relu').
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.intermediate(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "silu":
            x = F.silu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        x = self.dropout(x)
        x = self.output(x)
        return x

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.intermediate.weight)
        nn.init.xavier_uniform_(self.output.weight)
        if self.intermediate.bias is not None:
            nn.init.zeros_(self.intermediate.bias)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)


class GatedFeedForward(BaseNLPLayer):
    """
    Gated feed-forward network (used in LLaMA, PaLM, etc.).
    FFN_SwiGLU: output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Args:
        hidden_size: Model dimension.
        intermediate_size: Inner dimension of the FFN.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

    def init_weights(self) -> None:
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(proj.weight)
