"""
Tests for the visualization module.

Uses the 'Agg' backend to avoid display requirements.
"""

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import pytest

from visualization import WeightVisualizer, GradientVisualizer, AttentionVisualizer, ModelViewer


class _SimpleModel(nn.Module):
    """A minimal model for visualization tests."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class _TransformerBlock(nn.Module):
    """Minimal transformer block to produce attention weights."""
    def __init__(self, d_model=16, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return attn  # (B, num_heads, T, T)


class TestWeightVisualizer:
    """Test weight distribution visualization."""

    @pytest.fixture
    def model(self):
        return _SimpleModel()

    @pytest.fixture
    def viz(self, model):
        return WeightVisualizer(model)

    def test_extract_params(self, viz):
        params = viz._extract_params()
        assert len(params) >= 2, "Should find at least 2 weight tensors"
        assert any("fc1.weight" in k for k in params), "Should contain fc1.weight"

    def test_plot_weight_histograms(self, viz):
        fig = viz.plot_weight_histograms()
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_plot_weight_comparison(self, viz):
        fig = viz.plot_weight_comparison()
        assert fig is not None

    def test_plot_weight_heatmap(self, viz):
        fig = viz.plot_weight_heatmap(layer_name="fc1")
        assert fig is not None

    def test_plot_weight_heatmap_raises(self, viz):
        with pytest.raises(ValueError, match="No parameter found"):
            viz.plot_weight_heatmap(layer_name="nonexistent")

    def test_summary_statistics(self, viz):
        stats = viz.summary_statistics()
        assert len(stats) >= 2
        for key, s in stats.items():
            assert "mean" in s
            assert "std" in s
            assert "min" in s
            assert "max" in s
            assert "sparsity" in s
            assert "shape" in s

    def test_report(self, viz):
        report = viz.report()
        assert "STATISTICS REPORT" in report
        assert "fc1" in report or "fc2" in report

    def test_save_to_disk(self, viz, tmp_path):
        path = tmp_path / "weights.png"
        fig = viz.plot_weight_histograms(save_path=str(path))
        assert path.exists(), f"File {path} should exist"

    def test_report_save_to_disk(self, viz, tmp_path):
        path = tmp_path / "report.txt"
        viz.report(save_path=str(path))
        assert path.exists(), f"File {path} should exist"
        assert path.read_text().startswith("=")


class TestGradientVisualizer:
    """Test gradient distribution visualization."""

    @pytest.fixture
    def model(self):
        return _SimpleModel()

    @pytest.fixture
    def viz(self, model):
        return GradientVisualizer(model)

    def test_no_gradients_raises(self, viz):
        with pytest.raises(RuntimeError, match="No gradients"):
            viz.plot_gradient_histograms()

    def test_with_gradients(self, model):
        """Run a forward+backward pass to populate gradients."""
        x = torch.randn(4, 16)
        y = model(x).sum()
        y.backward()
        viz = GradientVisualizer(model)
        fig = viz.plot_gradient_histograms()
        assert fig is not None

    def test_gradient_flow(self, model):
        x = torch.randn(4, 16)
        y = model(x).sum()
        y.backward()
        viz = GradientVisualizer(model)
        fig = viz.plot_gradient_flow()
        assert fig is not None

    def test_track_and_timeline(self, model):
        viz = GradientVisualizer(model)
        for _ in range(5):
            x = torch.randn(4, 16)
            y = model(x).sum()
            y.backward()
            viz.track_step()
            model.zero_grad()

        assert len(viz._history) >= 1
        fig = viz.plot_gradient_timeline()
        assert fig is not None

    def test_reset_history(self, model):
        viz = GradientVisualizer(model)
        x = torch.randn(4, 16)
        y = model(x).sum()
        y.backward()
        viz.track_step()
        assert len(viz._history) >= 1
        viz.reset_history()
        assert len(viz._history) == 0


class TestAttentionVisualizer:
    """Test attention pattern visualization."""

    @pytest.fixture
    def block(self):
        return _TransformerBlock(d_model=16, num_heads=4)

    @pytest.fixture
    def attention_weights(self, block):
        x = torch.randn(1, 8, 16)  # (B, T, D)
        attn = block(x)
        return attn[0]  # (num_heads, T, T)

    @pytest.fixture
    def viz(self, block):
        return AttentionVisualizer(block)

    def test_plot_attention_heads(self, viz, attention_weights):
        fig = viz.plot_attention_heads(attention_weights, layer_name="test")
        assert fig is not None

    def test_attention_entropy(self, viz, attention_weights):
        entropy = viz.attention_entropy(attention_weights)
        assert entropy.shape[0] == 4, "Should have 4 entropy values"
        assert (entropy >= 0).all(), "Entropy should be non-negative"

    def test_plot_attention_entropy(self, viz, attention_weights):
        fig = viz.plot_attention_entropy(attention_weights, layer_name="test")
        assert fig is not None

    def test_attention_rollout(self, viz, attention_weights):
        # Simulate multiple layers with the same attention pattern
        layers = [attention_weights.clone() for _ in range(6)]
        fig = viz.plot_attention_rollout(layers, layer_step=2)
        assert fig is not None


class TestModelViewer:
    """Test the high-level ModelViewer API."""

    @pytest.fixture
    def model(self):
        return _SimpleModel()

    @pytest.fixture
    def viewer(self, model):
        return ModelViewer(model)

    def test_inspect_weights_histogram(self, viewer):
        fig = viewer.inspect_weights(plot_type="histogram")
        assert fig is not None

    def test_inspect_weights_comparison(self, viewer):
        fig = viewer.inspect_weights(plot_type="comparison")
        assert fig is not None

    def test_inspect_weights_heatmap(self, viewer):
        fig = viewer.inspect_weights(plot_type="heatmap", layer_name="fc1")
        assert fig is not None

    def test_inspect_weights_heatmap_no_layer_raises(self, viewer):
        with pytest.raises(ValueError, match="layer_name is required"):
            viewer.inspect_weights(plot_type="heatmap")

    def test_summary_report(self, viewer):
        report = viewer.summary_report()
        assert "STATISTICS REPORT" in report

    def test_inspect_gradients_flow(self, viewer):
        model = viewer.model
        x = torch.randn(4, 16)
        y = model(x).sum()
        y.backward()
        fig = viewer.inspect_gradients(plot_type="flow")
        assert fig is not None

    def test_inspect_gradients_histogram(self, viewer):
        model = viewer.model
        x = torch.randn(4, 16)
        y = model(x).sum()
        y.backward()
        fig = viewer.inspect_gradients(plot_type="histogram")
        assert fig is not None
