"""
Unit tests for nlp-nets layers.
"""

import torch
import pytest


class TestTokenEmbedding:
    """Tests for TokenEmbedding layer."""

    def test_forward_shape(self):
        from layers.embeddings import TokenEmbedding
        emb = TokenEmbedding(vocab_size=100, hidden_size=64)
        input_ids = torch.randint(0, 100, (2, 10))
        output = emb(input_ids)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_init_weights(self):
        from layers.embeddings import TokenEmbedding
        emb = TokenEmbedding(vocab_size=100, hidden_size=64, padding_idx=0)
        emb.init_weights()
        assert emb.embedding.weight[0].sum() == 0.0, "Padding idx weight should be zero"


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention layer."""

    def test_forward_shape(self):
        from layers.attention import MultiHeadAttention
        attn = MultiHeadAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_with_mask(self):
        from layers.attention import MultiHeadAttention
        attn = MultiHeadAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10).float()
        mask[:, :, :, 5:] = float("-inf")
        output, _ = attn(x, attention_mask=mask)
        assert output.shape == (2, 10, 64)


class TestPositionwiseFeedForward:
    """Tests for PositionwiseFeedForward layer."""

    def test_forward_shape(self):
        from layers.feedforward import PositionwiseFeedForward
        ffn = PositionwiseFeedForward(hidden_size=64, intermediate_size=256)
        x = torch.randn(2, 10, 64)
        output = ffn(x)
        assert output.shape == (2, 10, 64)

    def test_gelu_activation(self):
        from layers.feedforward import PositionwiseFeedForward
        ffn = PositionwiseFeedForward(hidden_size=64, intermediate_size=256, activation="gelu")
        x = torch.randn(2, 10, 64)
        output = ffn(x)
        assert not torch.isnan(output).any()


class TestGatedFeedForward:
    """Tests for GatedFeedForward layer."""

    def test_forward_shape(self):
        from layers.feedforward import GatedFeedForward
        ffn = GatedFeedForward(hidden_size=64, intermediate_size=256)
        x = torch.randn(2, 10, 64)
        output = ffn(x)
        assert output.shape == (2, 10, 64)


class TestLayerNorm:
    """Tests for normalization layers."""

    def test_layer_norm(self):
        from layers.normalization import LayerNorm
        ln = LayerNorm(64)
        x = torch.randn(2, 10, 64)
        output = ln(x)
        assert output.shape == (2, 10, 64)

    def test_rms_norm(self):
        from layers.normalization import RMSLayerNorm
        rms = RMSLayerNorm(64)
        x = torch.randn(2, 10, 64)
        output = rms(x)
        assert output.shape == (2, 10, 64)


class TestPositionalEncoding:
    """Tests for positional encoding layers."""

    def test_sinusoidal(self):
        from layers.positional_encoding import SinusoidalPositionalEncoding
        pe = SinusoidalPositionalEncoding(hidden_size=64, max_seq_length=100)
        x = torch.randn(2, 50, 64)
        output = pe(x)
        assert output.shape == (2, 50, 64)

    def test_rotary(self):
        from layers.positional_encoding import RotaryPositionalEncoding
        rope = RotaryPositionalEncoding(head_dim=32, max_seq_length=100)
        q = torch.randn(2, 8, 10, 32)
        k = torch.randn(2, 8, 10, 32)
        q_out, k_out = rope(q, k)
        assert q_out.shape == (2, 8, 10, 32)
        assert k_out.shape == (2, 8, 10, 32)

    def test_alibi(self):
        from layers.positional_encoding import ALiBiPositionalEncoding
        alibi = ALiBiPositionalEncoding(num_heads=8)
        bias = alibi(seq_len=10, device="cpu")
        assert bias.shape == (1, 8, 10, 10)
