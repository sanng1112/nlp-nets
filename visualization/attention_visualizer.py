"""
Attention pattern visualization.

Provides heatmaps and aggregated views of attention weight matrices
for interpreting model behavior and diagnosing attention collapse.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


class AttentionVisualizer:
    """
    Visualizes attention weight patterns from transformer models.

    Supports:
    - Per-head attention heatmaps
    - Head-aggregated attention patterns
    - Attention rollouts across layers
    - Attention entropy (confidence)  analysis
    """

    def __init__(
        self,
        model: nn.Module,
        figsize: Tuple[int, int] = (14, 10),
        style: str = "whitegrid",
        cmap: str = "Blues",
    ) -> None:
        """
        Args:
            model: A PyTorch transformer model with attention layers.
            figsize: Default figure size.
            style: Seaborn style.
            cmap: Colormap for attention heatmaps.
        """
        self.model = model
        self.figsize = figsize
        self.cmap = cmap
        sns.set_style(style)

    def plot_attention_heads(
        self,
        attention_weights: torch.Tensor,
        layer_name: str = "attention",
        n_cols: int = 4,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot individual attention head heatmaps.

        ``attention_weights`` should have shape
        ``(num_heads, seq_len, seq_len)`` or
        ``(batch, num_heads, seq_len, seq_len)`` (batch 0 is used).

        Args:
            attention_weights: Attention weight tensor from the model.
            layer_name: Label for the layer (used in titles).
            n_cols: Number of columns in the subplot grid.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        if attention_weights.ndim == 4:
            attention_weights = attention_weights[0]  # take first batch item

        num_heads, seq_len, _ = attention_weights.shape
        n_rows = (num_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows * 3))
        axes = axes.flatten() if num_heads > 1 else [axes]

        for h in range(num_heads):
            attn = attention_weights[h].detach().cpu().numpy()
            im = axes[h].imshow(attn, aspect="auto", cmap=self.cmap, vmin=0, vmax=attn.max())
            axes[h].set_title(f"Head {h}")
            axes[h].set_xlabel("Key positions")
            axes[h].set_ylabel("Query positions")
            plt.colorbar(im, ax=axes[h], shrink=0.7)

        for ax in axes[num_heads:]:
            ax.set_visible(False)

        fig.suptitle(f"Attention Heads — {layer_name}", fontsize=14)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_attention_rollout(
        self,
        attention_per_layer: List[torch.Tensor],
        layer_step: int = 1,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot an attention rollout heatmap showing how information
        propagates from input to output across layers.

        ``attention_per_layer`` is a list of tensors, each of shape
        ``(num_heads, seq_len, seq_len)`` or ``(seq_len, seq_len)``
        (averaged over heads).

        Args:
            attention_per_layer: Attention matrices from each layer.
            layer_step: Show every ``layer_step`` layers.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        n_layers = len(attention_per_layer)
        indices = list(range(0, n_layers, layer_step))
        n_plots = len(indices)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows * 3.5))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, layer_idx in enumerate(indices):
            attn = attention_per_layer[layer_idx]
            if attn.ndim == 3:
                attn = attn.mean(dim=0)  # average over heads
            attn_np = attn.detach().cpu().numpy()

            im = axes[i].imshow(attn_np, aspect="auto", cmap=self.cmap, vmin=0, vmax=attn_np.max())
            axes[i].set_title(f"Layer {layer_idx}")
            axes[i].set_xlabel("Key")
            axes[i].set_ylabel("Query")
            plt.colorbar(im, ax=axes[i], shrink=0.7)

        for ax in axes[n_plots:]:
            ax.set_visible(False)

        fig.suptitle("Attention Rollout Across Layers (head-averaged)", fontsize=14)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def attention_entropy(
        self, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the entropy of each attention head's distribution.

        Low entropy indicates sharp, confident attention patterns;
        high entropy indicates diffuse or uniform attention.

        Args:
            attention_weights: Tensor of shape ``(num_heads, seq_len, seq_len)``
                or ``(batch, num_heads, seq_len, seq_len)``.

        Returns:
            Entropy per head: shape ``(num_heads,)`` or ``(batch, num_heads)``.
        """
        if attention_weights.ndim == 4:
            batch, num_heads, seq_len, _ = attention_weights.shape
            # Average across batch and compute entropy
            attn = attention_weights.mean(dim=0)
        else:
            num_heads, seq_len, _ = attention_weights.shape
            attn = attention_weights

        # Avoid log(0)
        eps = 1e-8
        attn = attn.clamp(min=eps)
        entropy = -(attn * attn.log()).sum(dim=-1).mean(dim=-1)  # (num_heads,)
        return entropy.detach()

    def plot_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        layer_name: str = "attention",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Bar plot of attention entropy per head.

        Args:
            attention_weights: Tensor of shape ``(num_heads, seq_len, seq_len)``.
            layer_name: Label for the layer.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib Figure object.
        """
        entropy = self.attention_entropy(attention_weights)
        num_heads = entropy.shape[0]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("viridis", num_heads)
        ax.bar(range(num_heads), entropy.numpy(), color=colors)
        ax.set_xlabel("Attention head")
        ax.set_ylabel("Entropy (nat)")
        ax.set_title(f"Attention Head Entropy — {layer_name}")
        ax.set_xticks(range(num_heads))
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
