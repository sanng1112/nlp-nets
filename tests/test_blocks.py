"""
Unit tests for nlp-nets blocks.
"""

import torch
import pytest


class TestTransformerBlock:
    """Tests for the generic TransformerBlock."""

    @pytest.fixture
    def hidden_size(self):
        return 64

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def seq_len(self):
        return 10

    @pytest.fixture
    def x(self, batch_size, seq_len, hidden_size):
        return torch.randn(batch_size, seq_len, hidden_size)

    def _build_block(self, attention_cls, hidden_size, pre_norm=True, **attn_kwargs):
        from layers.feedforward import PositionwiseFeedForward
        from layers.normalization import LayerNorm
        from blocks.transformer_block import TransformerBlock

        attn = attention_cls(hidden_size=hidden_size, num_heads=8, **attn_kwargs)
        ffn = PositionwiseFeedForward(hidden_size=hidden_size, intermediate_size=hidden_size * 4)
        norm = LayerNorm(hidden_size)
        block = TransformerBlock(attn, ffn, norm, hidden_size=hidden_size, dropout=0.1, pre_norm=pre_norm)
        block.eval()
        return block

    def test_vanilla_attention(self, x):
        """TransformerBlock with vanilla MultiHeadAttention."""
        from layers.attention import MultiHeadAttention
        block = self._build_block(MultiHeadAttention, hidden_size=64)
        output = block(x)
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        assert torch.isfinite(output).all()

    def test_linear_attention(self, x):
        """TransformerBlock with LinearAttention."""
        from layers.linear_attention import LinearAttention
        block = self._build_block(LinearAttention, hidden_size=64)
        output = block(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_performer_attention(self, x):
        """TransformerBlock with PerformerAttention."""
        from layers.performer_attention import PerformerAttention
        block = self._build_block(PerformerAttention, hidden_size=64)
        output = block(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_cosformer_attention(self, x):
        """TransformerBlock with CosFormerAttention."""
        from layers.cosformer_attention import CosFormerAttention
        block = self._build_block(CosFormerAttention, hidden_size=64)
        output = block(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_nystrom_attention(self, x):
        """TransformerBlock with NystromAttention."""
        from layers.nystrom_attention import NystromAttention
        block = self._build_block(NystromAttention, hidden_size=64, num_landmarks=8)
        output = block(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_inla_attention(self, x):
        """TransformerBlock with INLAAttention."""
        from layers.inla_attention import INLAAttention
        block = self._build_block(INLAAttention, hidden_size=64)
        output = block(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_pre_ln_vs_post_ln(self, x):
        """Both Pre-LN and Post-LN should produce finite outputs."""
        from layers.attention import MultiHeadAttention
        pre = self._build_block(MultiHeadAttention, hidden_size=64, pre_norm=True)
        post = self._build_block(MultiHeadAttention, hidden_size=64, pre_norm=False)
        pre_out = pre(x)
        post_out = post(x)
        assert pre_out.shape == x.shape
        assert post_out.shape == x.shape
        assert torch.isfinite(pre_out).all()
        assert torch.isfinite(post_out).all()

    def test_with_mask(self, x):
        """Block should handle causal masks correctly."""
        from layers.attention import MultiHeadAttention
        block = self._build_block(MultiHeadAttention, hidden_size=64)
        mask = torch.zeros(1, 1, 1, x.size(1)).float()
        mask[:, :, :, x.size(1) // 2:] = float("-inf")
        output = block(x, attention_mask=mask)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, x):
        """Gradients should flow through the block."""
        from layers.attention import MultiHeadAttention
        from layers.feedforward import PositionwiseFeedForward
        from layers.normalization import LayerNorm
        from blocks.transformer_block import TransformerBlock

        attn = MultiHeadAttention(hidden_size=64, num_heads=8)
        ffn = PositionwiseFeedForward(hidden_size=64, intermediate_size=256)
        norm = LayerNorm(64)
        block = TransformerBlock(attn, ffn, norm, hidden_size=64)
        block.train()

        x.requires_grad_(True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_init_weights(self, x):
        """init_weights should not break the block."""
        from layers.attention import MultiHeadAttention
        block = self._build_block(MultiHeadAttention, hidden_size=64)
        block.init_weights()
        output = block(x)
        assert output.shape == x.shape
