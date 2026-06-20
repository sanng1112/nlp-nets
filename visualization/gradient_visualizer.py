"""
Gradient distribution visualization.

Tracks gradient statistics during training or after a backward pass
to diagnose vanishing/exploding gradients and dead neurons.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


class GradientVisualizer:
    """
    Visualizes gradient distributions across model layers.

    Supports:
    - Per-layer gradient histograms
    - Gradient statistics tracking over training steps
    - Gradient flow comparison across layers
    - Detection of dead parameters (zero gradients)
    """

    def __init__(
        self,
        model: nn.Module,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "whitegrid",
        palette: str = "mako",
    ) -> None:
        """
        Args:
            model: A PyTorch model whose gradients will be inspected.
            figsize: Default figure size.
            style: Seaborn style.
            palette: Color palette for gradient plots.
        """
        self.model = model
        self.figsize = figsize
        self.palette = palette
        self._history: Dict[str, List[float]] = defaultdict(list)
        sns.set_style(style)

    def _extract_grads(
        self, include_bias: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Extract named gradients that are not None."""
        grads: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if not include_bias and "bias" in name:
                continue
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu()
        return grads

    def plot_gradient_histograms(
        self,
        n_cols: int = 3,
        bins: int = 60,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot per-layer gradient histograms after a backward pass.

        Use this immediately after ``loss.backward()`` to inspect
        gradient health across the network.

        Args:
            n_cols: Number of columns in the subplot grid.
            bins: Number of histogram bins.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        grads = self._extract_grads()
        n_grads = len(grads)
        if n_grads == 0:
            raise RuntimeError(
                "No gradients found. Run loss.backward() before calling this method."
            )

        n_rows = (n_grads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows * 3.5))
        axes = axes.flatten() if n_grads > 1 else [axes]

        for ax, (name, grad) in zip(axes, grads.items()):
            g = grad.flatten().numpy()
            # Clip extreme outliers for better visualization
            lo, hi = np.percentile(g, [1, 99])
            g_clipped = g[(g >= lo) & (g <= hi)]

            ax.hist(g_clipped, bins=bins, alpha=0.7, density=True, color=sns.color_palette(self.palette)[0])
            ax.set_title(self._short_name(name), fontsize=9)
            ax.set_xlabel("Gradient value")
            ax.set_ylabel("Density")
            ax.axvline(0.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

            # Annotate fraction of zero gradients
            zero_frac = (g == 0).mean() * 100
            ax.annotate(
                f"Zero: {zero_frac:.1f}%",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=8,
                va="top",
                color="gray",
            )

        for ax in axes[n_grads:]:
            ax.set_visible(False)

        fig.suptitle("Per-Layer Gradient Distributions", fontsize=14, y=1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_gradient_flow(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot gradient mean and standard deviation per layer, showing
        how gradient magnitude changes through the network.

        Helps detect vanishing gradients (near-zero mean/stdev in early layers)
        or exploding gradients (very large values).

        Args:
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        grads = self._extract_grads()
        if len(grads) == 0:
            raise RuntimeError("No gradients found.")

        names = list(grads.keys())
        means = []
        stds = []
        for g in grads.values():
            g_np = g.numpy()
            means.append(np.mean(np.abs(g_np)))
            stds.append(np.std(g_np))

        fig, ax = plt.subplots(figsize=(self.figsize[0], 5))
        x = range(len(names))
        short_names = [self._short_name(n) for n in names]

        ax.bar(x, means, yerr=stds, capsize=3, color=sns.color_palette(self.palette, len(names)))
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean |gradient|")
        ax.set_title("Gradient Flow Across Layers")
        ax.set_yscale("log")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def track_step(self) -> None:
        """
        Record current gradient statistics for all layers.

        Call this after each optimizer ``step()`` to build a history
        that can be plotted with ``plot_gradient_timeline()``.
        """
        grads = self._extract_grads()
        for name, grad in grads.items():
            self._history[name].append(grad.abs().mean().item())

    def plot_gradient_timeline(
        self,
        layer_subset: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot mean |gradient| over training steps for selected layers.

        Args:
            layer_subset: List of layer name substrings to include.
                If None, includes all tracked layers.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        if not self._history:
            raise RuntimeError("No gradient history. Call track_step() during training.")

        fig, ax = plt.subplots(figsize=self.figsize)
        colors = sns.color_palette(self.palette, len(self._history))

        for (name, values), color in zip(self._history.items(), colors):
            if layer_subset and not any(l in name for l in layer_subset):
                continue
            ax.plot(values, label=self._short_name(name), color=color, alpha=0.8, linewidth=1.2)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean |gradient|")
        ax.set_title("Gradient Magnitude Over Time")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def reset_history(self) -> None:
        """Clear all tracked gradient history."""
        self._history.clear()

    @staticmethod
    def _short_name(name: str) -> str:
        for prefix in ["model.", "module.", "_orig_mod."]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name if len(name) <= 50 else "..." + name[-47:]
