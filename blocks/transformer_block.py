"""
Generic transformer block with pluggable attention, feed-forward, and
normalization layers.

Supports both Pre-LN (default) and Post-LN residual configurations.

Usage:
    attn = MultiHeadAttention(hidden_size=512, num_heads=8)
    ffn = PositionwiseFeedForward(hidden_size=512, intermediate_size=2048)
    norm = LayerNorm(512)

    block = TransformerBlock(attn, ffn, norm, dropout=0.1)
    output = block(x, attention_mask=mask)
"""

from typing import Optional

from torch import Tensor, nn

from blocks.base_block import BaseTransformerBlock


class TransformerBlock(BaseTransformerBlock):
    """
    A standard transformer block with configurable attention, FFN, and norm.

    Pre-LN (default):
        x = x + attn(norm_attn(x))
        x = x + ffn(norm_ffn(x))

    Post-LN:
        x = norm_attn(x + attn(x))
        x = norm_ffn(x + ffn(x))

    Args:
        attention_layer: Attention module (see layers.attention, layers.*_attention).
        ffn_layer: Feed-forward module (see layers.feedforward).
        norm_layer: Normalization module (see layers.normalization).
            If None, LayerNorm with hidden_size will be created. Must be callable
            or a factory with signature (hidden_size) -> Module.
        hidden_size: Hidden size (required if norm_layer is None).
        dropout: Dropout probability applied after both attention and FFN.
        pre_norm: If True (default), use Pre-LN; otherwise Post-LN.
    """

    def __init__(
        self,
        attention_layer: nn.Module,
        ffn_layer: nn.Module,
        norm_layer: Optional[nn.Module] = None,
        hidden_size: Optional[int] = None,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.attention = attention_layer
        self.ffn = ffn_layer
        self.pre_norm = pre_norm
        self.dropout = nn.Dropout(dropout)

        # Infer hidden_size from attention module if not provided
        if hidden_size is None:
            hidden_size = getattr(attention_layer, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(
                    "hidden_size must be provided if attention_layer does not expose it."
                )

        # Create normalization layers
        if norm_layer is None:
            from layers.normalization import LayerNorm

            self.norm_attn = LayerNorm(hidden_size)
            self.norm_ffn = LayerNorm(hidden_size)
        else:
            self.norm_attn = norm_layer
            self.norm_ffn = (
                norm_layer.__class__(hidden_size) if hasattr(norm_layer, "__class__") else norm_layer
            )
            # If norm_layer is a single instance, create a second one for FFN
            if self.norm_attn is self.norm_ffn:
                self.norm_ffn = norm_layer.__class__(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: Optional (batch_size, 1, seq_len, seq_len) mask.

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        if self.pre_norm:
            # Pre-LN: norm → sublayer → residual
            attn_output, *_ = self.attention(self.norm_attn(hidden_states), attention_mask=attention_mask)
            hidden_states = hidden_states + self.dropout(attn_output)

            ffn_output = self.ffn(self.norm_ffn(hidden_states))
            hidden_states = hidden_states + self.dropout(ffn_output)
        else:
            # Post-LN: sublayer → residual → norm
            attn_output, *_ = self.attention(hidden_states, attention_mask=attention_mask)
            hidden_states = self.norm_attn(hidden_states + self.dropout(attn_output))

            ffn_output = self.ffn(hidden_states)
            hidden_states = self.norm_ffn(hidden_states + self.dropout(ffn_output))

        return hidden_states

    def init_weights(self) -> None:
        for module in [self.attention, self.ffn, self.norm_attn, self.norm_ffn]:
            if hasattr(module, "init_weights"):
                module.init_weights()
