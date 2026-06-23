"""
Configuration module for nlp-nets.

Provides:
- ``ConfigResolver``: dotted-key access to nested config dicts.
- ``ConfigSchema``: validation helpers for model/train/optim configs.
- ``ConfigValidationError``: raised when validation fails.
"""

from config.resolver import ConfigResolver
from config.schema import ConfigSchema, ConfigValidationError

__all__ = [
    "ConfigResolver",
    "ConfigSchema",
    "ConfigValidationError",
]
