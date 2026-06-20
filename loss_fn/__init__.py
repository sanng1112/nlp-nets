#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 nlp-nets. All Rights Reserved.
#

import argparse
from typing import Optional

from loss_fn.base_criteria import BaseCriteria
from utils import logger
from utils.registry import Registry


LOSS_REGISTRY = Registry(
    registry_name="loss_functions",
    base_class=BaseCriteria,
    lazy_load_dirs=["loss_fn"],
    internal_dirs=["internal", "internal/projects/*"],
)


def build_loss_fn(opts: argparse.Namespace, category: Optional[str] = "", *args, **kwargs) -> BaseCriteria:
    """
    Build a loss function from configuration.

    Args:
        opts: Configuration dict or argparse namespace.
        category: Loss category (e.g., 'mlm', 'classification').
        *args, **kwargs: Additional arguments passed to the loss constructor.

    Returns:
        An initialized loss function instance.
    """
    if not category:
        category = getattr(opts, "loss.category", opts.get("task", {}).get("loss", None))

    if category is None:
        logger.error(
            "Please specify loss name using --loss.category or task.loss in config."
        )

    if isinstance(opts, dict):
        loss_name = opts.get("loss", {}).get("name", category)
    else:
        loss_name = getattr(opts, f"loss.{category}.name", category)

    if loss_name == "__base__":
        logger.error("__base__ can't be used as a loss function name. Please check.")

    loss_fn = LOSS_REGISTRY[loss_name, category](opts, *args, **kwargs)
    return loss_fn


def add_loss_fn_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add loss function CLI arguments."""
    parser = BaseCriteria.add_arguments(parser=parser)
    parser = LOSS_REGISTRY.all_arguments(parser)
    return parser
