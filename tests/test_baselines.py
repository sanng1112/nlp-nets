"""
Unit tests for all baseline attention mechanisms and INLA.
Tests forward shape, masking behavior, and weight initialization.
"""

import torch
import pytest


class TestLinearAttention:
    """Tests for LinearAttention (Katharopoulos)."""

    def test_forward_shape(self):
        from layers.linear_attention import LinearAttention
        attn = LinearAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_with_causal_mask(self):
        from layers.linear_attention import LinearAttention
        attn = LinearAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10).float()
        mask[:, :, :, 5:] = float("-inf")
        output, _ = attn(x, attention_mask=mask)
        assert output.shape == (2, 10, 64)

    def test_init_weights(self):
        from layers.linear_attention import LinearAttention
        attn = LinearAttention(hidden_size=64, num_heads=8)
        attn.init_weights()
        # Check no NaN after init
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert not torch.isnan(output).any()

    def test_self_attention_stable(self):
        """Linear attention should not produce NaN for random inputs."""
        from layers.linear_attention import LinearAttention
        attn = LinearAttention(hidden_size=32, num_heads=4, eps=1e-8)
        for _ in range(5):
            x = torch.randn(2, 10, 32)
            out, _ = attn(x)
            assert torch.isfinite(out).all()


class TestPerformerAttention:
    """Tests for PerformerAttention (FAVOR+)."""

    def test_forward_shape(self):
        from layers.performer_attention import PerformerAttention
        attn = PerformerAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_forward_softmax_kernel(self):
        from layers.performer_attention import PerformerAttention
        attn = PerformerAttention(hidden_size=64, num_heads=8, kernel_type="softmax")
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64)

    def test_forward_relu_kernel(self):
        from layers.performer_attention import PerformerAttention
        attn = PerformerAttention(hidden_size=64, num_heads=8, kernel_type="relu")
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64)

    def test_init_weights(self):
        from layers.performer_attention import PerformerAttention
        attn = PerformerAttention(hidden_size=64, num_heads=8)
        attn.init_weights()
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert not torch.isnan(output).any()

    def test_different_num_random_features(self):
        from layers.performer_attention import PerformerAttention
        attn = PerformerAttention(hidden_size=64, num_heads=8, num_random_features=16)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64)

    def test_causal_mask(self):
        from layers.performer_attention import PerformerAttention
        attn = PerformerAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10).float()
        mask[:, :, :, 5:] = float("-inf")
        output, _ = attn(x, attention_mask=mask)
        assert output.shape == (2, 10, 64)


class TestNystromAttention:
    """Tests for NystromAttention (Nyströmformer)."""

    def test_forward_shape(self):
        from layers.nystrom_attention import NystromAttention
        attn = NystromAttention(hidden_size=64, num_heads=8, num_landmarks=8)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_short_sequence(self):
        """Sequence shorter than num_landmarks should still work."""
        from layers.nystrom_attention import NystromAttention
        attn = NystromAttention(hidden_size=64, num_heads=8, num_landmarks=16)
        x = torch.randn(2, 5, 64)
        output, _ = attn(x)
        assert output.shape == (2, 5, 64)

    def test_init_weights(self):
        from layers.nystrom_attention import NystromAttention
        attn = NystromAttention(hidden_size=64, num_heads=8)
        attn.init_weights()
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert not torch.isnan(output).any()

    def test_pseudo_inverse_stable(self):
        """Nyström pseudo-inverse should be numerically stable for square matrices."""
        from layers.nystrom_attention import NystromAttention
        # Generate a batch of square matrices (landmark × landmark)
        # Use a diagonally-dominant matrix for better conditioning
        x = torch.randn(4, 8, 8) * 0.1
        x = x + torch.eye(8).unsqueeze(0) * 3.0  # add diagonal dominance
        x_inv = NystromAttention._pseudo_inverse(x)
        assert x_inv.shape == (4, 8, 8)
        # x @ x_inv ≈ I
        approx_identity = x @ x_inv
        identity = torch.eye(8).unsqueeze(0).expand(4, -1, -1)
        error = (approx_identity - identity).abs().mean()
        assert error < 0.05, f"Pseudo-inverse reconstruction error too high: {error}"


class TestCosFormerAttention:
    """Tests for CosFormerAttention."""

    def test_forward_shape(self):
        from layers.cosformer_attention import CosFormerAttention
        attn = CosFormerAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_init_weights(self):
        from layers.cosformer_attention import CosFormerAttention
        attn = CosFormerAttention(hidden_size=64, num_heads=8)
        attn.init_weights()
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert not torch.isnan(output).any()

    def test_causal_mask(self):
        from layers.cosformer_attention import CosFormerAttention
        attn = CosFormerAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10).float()
        mask[:, :, :, 5:] = float("-inf")
        output, _ = attn(x, attention_mask=mask)
        assert output.shape == (2, 10, 64)

    def test_cosine_weights_precomputed(self):
        from layers.cosformer_attention import CosFormerAttention
        attn = CosFormerAttention(hidden_size=64, num_heads=8, max_seq_length=512)
        assert attn.cos_weights.shape == (512,)
        assert attn.cos_weights[0] == 1.0  # cos(0) = 1


class TestINLAAttention:
    """Tests for INLAAttention (proposed)."""

    def test_forward_shape(self):
        from layers.inla_attention import INLAAttention
        attn = INLAAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_init_weights(self):
        from layers.inla_attention import INLAAttention
        attn = INLAAttention(hidden_size=64, num_heads=8)
        attn.init_weights()
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert not torch.isnan(output).any()

    def test_causal_mask(self):
        from layers.inla_attention import INLAAttention
        attn = INLAAttention(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10).float()
        mask[:, :, :, 5:] = float("-inf")
        output, _ = attn(x, attention_mask=mask)
        assert output.shape == (2, 10, 64)

    def test_lifting_output_dim(self):
        from layers.inla_attention import INLALifting
        lifting = INLALifting(d=64, d_k=16, r=128)
        x = torch.randn(2, 10, 64)
        out = lifting(x)
        assert out.shape == (2, 10, 128), f"Expected (2, 10, 128), got {out.shape}"

    def test_lifting_different_activations(self):
        from layers.inla_attention import INLALifting
        for act in ("gelu", "silu", "relu"):
            lifting = INLALifting(d=64, activation=act)
            x = torch.randn(2, 10, 64)
            out = lifting(x)
            assert out.shape == (2, 10, 128)
            assert torch.isfinite(out).all()

    def test_custom_r_d_k(self):
        from layers.inla_attention import INLAAttention
        attn = INLAAttention(hidden_size=64, num_heads=8, r=32, d_k=4)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64)
