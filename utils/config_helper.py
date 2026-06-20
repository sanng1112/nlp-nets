import os
from typing import Any, Optional, Dict

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
