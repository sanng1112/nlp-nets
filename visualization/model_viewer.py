"""
High-level model inspection API.

Combines weight, gradient, and attention visualizers into a single
convenient interface for post-training model analysis.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch.nn as nn

from visualization.attention_visualizer import AttentionVisualizer
from visualization.gradient_visualizer import GradientVisualizer
from visualization.weight_visualizer import WeightVisualizer


class ModelViewer:
    """
    Unified API for inspecting a trained model's internals.

    Provides one-line access to:
    - Weight distribution histograms and heatmaps
    - Gradient distribution and flow charts
    - Attention head patterns and entropy
    - Summary statistics reports
    """

    def __init__(
        self,
        model: nn.Module,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "whitegrid",
    ) -> None:
        """
        Args:
            model: A trained PyTorch model.
            figsize: Default figure size for plots.
            style: Seaborn style ('whitegrid', 'darkgrid', 'ticks', etc.).
        """
        self.model = model
        self.weight_viz = WeightVisualizer(model, figsize=figsize, style=style)
        self.gradient_viz = GradientVisualizer(model, figsize=figsize, style=style)
        self.attention_viz = AttentionVisualizer(model, figsize=figsize, style=style)

    def inspect_weights(
        self,
        plot_type: str = "histogram",
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Quick weight inspection.

        Args:
            plot_type: One of ``'histogram'`` (grid of per-layer histograms),
                ``'comparison'`` (overlayed KDE), or ``'heatmap'``
                (weight matrix heatmap for a specific layer).
            layer_name: Required for ``plot_type='heatmap'``.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        if plot_type == "histogram":
            return self.weight_viz.plot_weight_histograms(save_path=save_path)
        elif plot_type == "comparison":
            return self.weight_viz.plot_weight_comparison(save_path=save_path)
        elif plot_type == "heatmap":
            if layer_name is None:
                raise ValueError("layer_name is required for heatmap plot_type.")
            return self.weight_viz.plot_weight_heatmap(layer_name, save_path=save_path)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Choose from: histogram, comparison, heatmap.")

    def inspect_gradients(
        self,
        plot_type: str = "histogram",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Quick gradient inspection (call after ``loss.backward()``).

        Args:
            plot_type: ``'histogram'`` (per-layer gradients) or
                ``'flow'`` (gradient magnitude across layers).
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        if plot_type == "histogram":
            return self.gradient_viz.plot_gradient_histograms(save_path=save_path)
        elif plot_type == "flow":
            return self.gradient_viz.plot_gradient_flow(save_path=save_path)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}.")

    def inspect_attention(
        self,
        attention_weights: torch.Tensor,
        layer_name: str = "attention",
        plot_type: str = "heads",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Quick attention inspection.

        Args:
            attention_weights: Tensor of shape ``(num_heads, seq_len, seq_len)``.
            layer_name: Label for the layer.
            plot_type: ``'heads'`` (per-head heatmaps), ``'entropy'``
                (entropy bar chart).
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        if plot_type == "heads":
            return self.attention_viz.plot_attention_heads(
                attention_weights, layer_name=layer_name, save_path=save_path
            )
        elif plot_type == "entropy":
            return self.attention_viz.plot_attention_entropy(
                attention_weights, layer_name=layer_name, save_path=save_path
            )
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}.")

    def summary_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a full text report of model weight statistics.

        Args:
            save_path: Optional file path to save the report.

        Returns:
            The formatted report string.
        """
        return self.weight_viz.report(save_path=save_path)
