import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from layers.base_layer import BaseNLPLayer


class TokenEmbedding(BaseNLPLayer):
    """
    Token embeddings (lookup table).

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Embedding dimension.
        padding_idx: Index for padding token (if any).
    """

    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.hidden_size = hidden_size

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids) * math.sqrt(self.hidden_size)

    def init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0.0)


class PositionalEmbedding(BaseNLPLayer):
    """
    Learnable positional embeddings.

    Args:
        max_seq_length: Maximum sequence length.
        hidden_size: Embedding dimension.
    """

    def __init__(self, max_seq_length: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, hidden_size)

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.embedding(positions)

    def init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)


class SegmentEmbedding(BaseNLPLayer):
    """
    Learnable segment (token type) embeddings, used in BERT.

    Args:
        num_segments: Number of segments (typically 2).
        hidden_size: Embedding dimension.
    """

    def __init__(self, num_segments: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_segments, hidden_size)

    def forward(self, token_type_ids: Tensor) -> Tensor:
        return self.embedding(token_type_ids)

    def init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)


class BERTEmbeddings(BaseNLPLayer):
    """
    BERT-style embeddings: Token + Position + Segment.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Embedding dimension.
        max_seq_length: Maximum sequence length.
        num_segments: Number of segments (default: 2).
        dropout: Dropout probability.
        padding_idx: Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_length: int = 512,
        num_segments: int = 2,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token = TokenEmbedding(vocab_size, hidden_size, padding_idx)
        self.position = PositionalEmbedding(max_seq_length, hidden_size)
        self.segment = SegmentEmbedding(num_segments, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        seq_len = input_ids.size(1)
        token_emb = self.token(input_ids)

        pos_emb = self.position(seq_len, input_ids.device)
        if position_ids is not None:
            pos_emb = self.embedding(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        seg_emb = self.segment(token_type_ids)

        embeddings = token_emb + pos_emb + seg_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def init_weights(self) -> None:
        self.token.init_weights()
        self.position.init_weights()
        self.segment.init_weights()
