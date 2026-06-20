"""
Optimizer builder — constructs PyTorch optimizers from config.
"""

from typing import Any, Dict

import torch
import torch.optim as optim


def add_optimizer_args(parser: Any) -> Any:
    """Add optimizer CLI arguments (kept for compatibility)."""
    return parser


def build_optimizer(opts: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from configuration.

    Args:
        opts: Configuration dictionary (typically loaded from YAML).
        model: The model whose parameters will be optimized.

    Returns:
        A PyTorch optimizer instance.
    """
    optim_config = opts.get("optim", {})
    optim_name = optim_config.get("name", "adamw").lower()
    lr = optim_config.get("lr", 1e-4)
    weight_decay = optim_config.get("weight_decay", 0.01)
    beta1 = optim_config.get("beta1", 0.9)
    beta2 = optim_config.get("beta2", 0.999)
    eps = optim_config.get("eps", 1e-8)
    momentum = optim_config.get("momentum", 0.9)

    # Apply weight decay only to non-bias/norm parameters
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name or "layernorm" in name or "ln_" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optim_name == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optim_name == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optim_name == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optim_name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                param_groups,
                lr=lr,
                scale_parameter=True,
                relative_step=False,
            )
        except ImportError:
            raise ImportError("transformers package required for Adafactor. Install: pip install transformers")
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    return optimizer
