"""
nlp-nets blocks: Composable transformer building blocks with pluggable attention.

Each block wraps an attention mechanism, feed-forward network, and normalization
into a standard Pre-LN or Post-LN transformer block.
"""

from blocks.base_block import BaseTransformerBlock
from blocks.transformer_block import TransformerBlock

__all__ = [
    "BaseTransformerBlock",
    "TransformerBlock",
]
