"""
Weight distribution visualization for post-training model analysis.

Provides histograms, density plots, and summary statistics for
weight matrices across all parameterized layers in a model.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from layers.base_layer import BaseNLPLayer


class WeightVisualizer:
    """
    Visualizes weight distributions across all parameterized layers of a model.

    Supports:
    - Per-layer weight histograms
    - Combined multi-layer comparison plots
    - Heatmaps of weight matrices
    - Statistical summaries (mean, std, min, max, sparsity)
    """

    def __init__(
        self,
        model: nn.Module,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "whitegrid",
        palette: str = "viridis",
    ) -> None:
        """
        Args:
            model: A PyTorch model (trained or untrained) to inspect.
            figsize: Default figure size for plots.
            style: Seaborn style (e.g., 'whitegrid', 'darkgrid', 'ticks').
            palette: Color palette for distribution plots.
        """
        self.model = model
        self.figsize = figsize
        self.palette = palette
        sns.set_style(style)

    def _extract_params(
        self, include_bias: bool = True, include_embedding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract named parameters from the model, optionally filtering
        by type.

        Returns:
            Dictionary mapping parameter names to their weight tensors.
        """
        params: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if not include_bias and "bias" in name:
                continue
            if not include_embedding and "embedding" in name:
                continue
            params[name] = param.detach().cpu()
        return params

    def plot_weight_histograms(
        self,
        n_cols: int = 3,
        bins: int = 80,
        share_x: bool = False,
        show_outliers: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot per-layer weight histograms in a grid layout.

        Each subplot shows the distribution of weight values for one
        parameter tensor, allowing visual inspection of initialization
        quality, convergence, or vanishing/exploding patterns.

        Args:
            n_cols: Number of columns in the subplot grid.
            bins: Number of histogram bins.
            share_x: Whether to share the x-axis across subplots.
            show_outliers: If True, shows the full value range.
            save_path: Optional file path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        params = self._extract_params()
        n_params = len(params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows * 3.5))
        axes = axes.flatten() if n_params > 1 else [axes]

        for ax, (name, weight) in zip(axes, params.items()):
            w = weight.flatten().numpy()
            if not show_outliers:
                lo, hi = np.percentile(w, [1, 99])
                w = w[(w >= lo) & (w <= hi)]

            ax.hist(w, bins=bins, alpha=0.7, density=True, color=sns.color_palette(self.palette)[0])
            ax.set_title(self._short_name(name), fontsize=10)
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Density")
            ax.axvline(0.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

        # Hide unused subplots
        for ax in axes[n_params:]:
            ax.set_visible(False)

        fig.suptitle("Per-Layer Weight Distributions", fontsize=14, y=1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_weight_comparison(
        self,
        layer_names: Optional[List[str]] = None,
        bins: int = 80,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Overlay multiple weight histograms on a single plot for
        easy cross-layer comparison.

        Args:
            layer_names: Specific layer names to include. If None,
                shows all weight layers.
            bins: Number of histogram bins.
            save_path: Optional file path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        params = self._extract_params()
        if layer_names:
            params = {k: v for k, v in params.items() if any(l in k for l in layer_names)}

        fig, ax = plt.subplots(figsize=self.figsize)
        colors = sns.color_palette(self.palette, len(params))

        for (name, weight), color in zip(params.items(), colors):
            w = weight.flatten().numpy()
            sns.kdeplot(w, ax=ax, label=self._short_name(name), color=color, bw_adjust=0.5)

        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        ax.set_title("Weight Distribution Comparison Across Layers")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_weight_heatmap(
        self,
        layer_name: str,
        max_neurons: int = 64,
        cmap: str = "RdBu_r",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Display a heatmap of a specific weight matrix.

        Useful for detecting dead neurons, rank collapse, or
        structured sparsity patterns.

        Args:
            layer_name: Substring to match the target parameter name.
            max_neurons: Maximum rows/cols to display (samples uniformly).
            cmap: Matplotlib colormap.
            save_path: Optional file path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        params = self._extract_params()
        target = None
        target_name = None
        for name, weight in params.items():
            if layer_name in name:
                target = weight
                target_name = name
                break

        if target is None:
            raise ValueError(f"No parameter found matching '{layer_name}'. Available: {list(params.keys())}")

        w = target.numpy()
        if w.ndim > 2:
            w = w.reshape(w.shape[0], -1)

        # Down-sample if too large
        if w.shape[0] > max_neurons:
            idx = np.linspace(0, w.shape[0] - 1, max_neurons, dtype=int)
            w = w[idx, :]
        if w.shape[1] > max_neurons:
            idx = np.linspace(0, w.shape[1] - 1, max_neurons, dtype=int)
            w = w[:, idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(w, aspect="auto", cmap=cmap, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Weight Heatmap: {self._short_name(target_name)}")
        ax.set_xlabel("Input features (sampled)")
        ax.set_ylabel("Output features (sampled)")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics (mean, std, min, max, sparsity)
        for each weight tensor in the model.

        Returns:
            Nested dict mapping parameter names to their statistics.
        """
        params = self._extract_params()
        stats: Dict[str, Dict[str, float]] = {}
        for name, weight in params.items():
            w = weight.flatten()
            stats[self._short_name(name)] = {
                "mean": float(w.mean().item()),
                "std": float(w.std().item()),
                "min": float(w.min().item()),
                "max": float(w.max().item()),
                "sparsity": float((w.abs() < 1e-8).float().mean().item()) * 100,
                "shape": list(weight.shape),
            }
        return stats

    def report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a human-readable text report of all parameter statistics.

        Args:
            save_path: Optional file path to save the report.

        Returns:
            The formatted report string.
        """
        stats = self.summary_statistics()
        lines = ["=" * 72, "  POST-TRAINING WEIGHT STATISTICS REPORT", "=" * 72, ""]
        lines.append(f"{'Layer':<40} {'Shape':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Sparsity%':<10}")
        lines.append("-" * 110)
        for name, s in stats.items():
            shape_str = "x".join(str(d) for d in s["shape"])
            lines.append(
                f"{name:<40} {shape_str:<20} {s['mean']:<10.4f} {s['std']:<10.4f} "
                f"{s['min']:<10.4f} {s['max']:<10.4f} {s['sparsity']:<10.2f}"
            )
        lines.append("=" * 110)

        report = "\n".join(lines)
        if save_path:
            with open(save_path, "w") as f:
                f.write(report)

        print(report)
        return report

    @staticmethod
    def _short_name(name: str) -> str:
        """Shorten a parameter name for display in titles."""
        # Remove common prefixes
        for prefix in ["model.", "module.", "_orig_mod."]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name if len(name) <= 50 else "..." + name[-47:]
