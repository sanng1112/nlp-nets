"""
nlp-nets visualization: Post-training model inspection and plotting utilities.

Provides tools to visualize weight distributions, gradient flow,
and attention patterns for diagnosing model health after training.

Usage:
    from visualization import ModelViewer

    viewer = ModelViewer(model)
    viewer.inspect_weights(plot_type="histogram")     # weight histograms
    viewer.inspect_weights(plot_type="heatmap", ...)   # weight heatmap
    viewer.inspect_gradients(plot_type="flow")         # gradient flow
    viewer.inspect_attention(attn_w, plot_type="heads") # attention heads
    viewer.summary_report()                            # text statistics
"""

from visualization.weight_visualizer import WeightVisualizer
from visualization.gradient_visualizer import GradientVisualizer
from visualization.attention_visualizer import AttentionVisualizer
from visualization.model_viewer import ModelViewer

__all__ = [
    "WeightVisualizer",
    "GradientVisualizer",
    "AttentionVisualizer",
    "ModelViewer",
]
