import os
import argparse
from typing import Any, Optional, Dict, Tuple

import yaml


def get_param(
    opts: Any,
    explicit_val: Optional[Any] = None,
    attr_name: str = None,
    default_val: Optional[Any] = None,
) -> Any:
    """
    Generic parameter retrieval with priority:
    Explicit value > opts dict value > default value.
    """
    if explicit_val is not None:
        return explicit_val
    if opts is not None and isinstance(opts, dict):
        # Support nested keys like "model.name"
        keys = attr_name.split(".") if attr_name else []
        val = opts
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default_val
        return val if val is not None else default_val
    return default_val


def parse_arguments() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Parse CLI arguments and load YAML config.

    Returns:
        Tuple of (args_namespace, config_dict).
    """
    parser = argparse.ArgumentParser(description="nlp-nets Training Entry Point")
    parser.add_argument("--common.config-file", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--common.sanity-check", action="store_true", help="Run sanity check only")
    parser.add_argument("--common.resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--common.eval-only", action="store_true", help="Run evaluation only (no training)")

    # CLI overrides for key config values
    parser.add_argument("--model.name", type=str, default=None, help="Model name override")
    parser.add_argument("--dataset.name", type=str, default=None, help="Dataset name override")
    parser.add_argument("--optim.lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--train.batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--train.epochs", type=int, default=None, help="Number of epochs override")
    parser.add_argument("--train.device", type=str, default=None, help="Device override (cuda/cpu)")

    args = parser.parse_args()
    opts = load_config(args.common_config_file)

    # Apply CLI overrides
    if getattr(args, "model_name", None) is not None:
        opts["model"]["name"] = args.model_name
    if getattr(args, "dataset_name", None) is not None:
        opts["dataset"]["name"] = args.dataset_name
    if getattr(args, "optim_lr", None) is not None:
        opts["optim"]["lr"] = args.optim_lr
    if getattr(args, "train_batch_size", None) is not None:
        opts["train"]["batch_size"] = args.train_batch_size
    if getattr(args, "train_epochs", None) is not None:
        opts["train"]["epochs"] = args.train_epochs
    if getattr(args, "train_device", None) is not None:
        opts["train"]["device"] = args.train_device

    opts["common"] = {
        "config_file": args.common_config_file,
        "sanity_check": getattr(args, "common_sanity_check", False),
        "resume": getattr(args, "common_resume", None),
        "eval_only": getattr(args, "common_eval_only", False),
    }

    return args, opts


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dict with configuration values.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Resolve ${ENV_VAR} placeholders
    def _resolve_env(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        return value

    def _deep_resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _deep_resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deep_resolve(v) for v in obj]
        return _resolve_env(obj)

    config = _deep_resolve(config)
    return config
