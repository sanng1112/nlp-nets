"""
LR Scheduler builder — constructs PyTorch LR schedulers from config.
"""

import argparse
from typing import Any, Dict, Optional

import torch
import torch.optim.lr_scheduler as lr_scheduler


def add_scheduler_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add scheduler CLI arguments (kept for compatibility)."""
    return parser


def build_scheduler(
    opts: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    num_training_steps: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Build a learning rate scheduler from configuration.

    Args:
        opts: Configuration dictionary.
        optimizer: The optimizer to schedule.
        num_training_steps: Total number of training steps (required for some schedulers).

    Returns:
        A PyTorch LR scheduler, or None if no scheduler is specified.
    """
    scheduler_config = opts.get("optim", {}).get("scheduler", {})
    if not scheduler_config:
        return None

    scheduler_name = scheduler_config.get("name", "linear_warmup").lower()
    warmup_ratio = scheduler_config.get("warmup_ratio", 0.06)
    warmup_steps = scheduler_config.get("warmup_steps", 0)
    num_epochs = opts.get("train", {}).get("epochs", 3)

    # Prefer warmup_steps over warmup_ratio
    if warmup_steps == 0 and num_training_steps is not None:
        warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_name == "linear_warmup":
        if num_training_steps is None:
            return None
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "cosine":
        t_max = scheduler_config.get("t_max", num_epochs)
        eta_min = scheduler_config.get("min_lr", 0.0)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif scheduler_name == "cosine_warmup":
        if num_training_steps is None:
            return None
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "polynomial":
        if num_training_steps is None:
            return None
        from transformers import get_polynomial_decay_schedule_with_warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            power=scheduler_config.get("power", 1.0),
        )
    elif scheduler_name == "constant":
        from transformers import get_constant_schedule
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_name == "constant_warmup":
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler_name == "reduce_on_plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.1),
            patience=scheduler_config.get("patience", 10),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler
