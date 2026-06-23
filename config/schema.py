"""
ConfigSchema — validation helpers for nlp-nets configuration dictionaries.

Provides static methods that validate the structure of model and training
configuration dicts, raising ``ConfigValidationError`` on failure.

This module is part of the nlp-nets configuration system
with NLP-specific additions.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Custom Error
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Base exception for configuration-related errors."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when a configuration dictionary fails schema validation."""

    pass


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_REQUIRED_MODEL_KEYS: List[str] = ["name", "vocab_size", "hidden_size"]
_REQUIRED_TRAIN_KEYS: List[str] = ["batch_size", "epochs"]
_REQUIRED_OPTIM_KEYS: List[str] = ["name", "lr"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConfigSchema:
    """Static validation methods for configuration dictionaries."""

    # -- Top-level ----------------------------------------------------------

    @staticmethod
    def validate_top_level(config: Dict[str, Any]) -> None:
        """Validate that the config has at least one of the known sections.

        Parameters
        ----------
        config : dict
            The full configuration dictionary.

        Raises
        ------
        ConfigValidationError
            If none of the expected top-level keys exist.
        """
        if not isinstance(config, dict):
            raise ConfigValidationError(
                f"Configuration must be a dict, got {type(config).__name__!r}."
            )

    # -- Model section ------------------------------------------------------

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> None:
        """Validate the model configuration section.

        Checks:
            1. A ``"model"`` section exists in *config*.
            2. The ``"model"`` section is a dict.
            3. It contains required keys: ``name``, ``vocab_size``, ``hidden_size``.

        Parameters
        ----------
        config : dict
            The full configuration dictionary.

        Raises
        ------
        ConfigValidationError
            If any of the checks fail.
        """
        ConfigSchema.validate_top_level(config)

        if "model" not in config:
            raise ConfigValidationError(
                "Configuration is missing the required 'model' section."
            )
        model_cfg = config["model"]
        if not isinstance(model_cfg, dict):
            raise ConfigValidationError(
                f"The 'model' section must be a dict, got "
                f"{type(model_cfg).__name__!r}."
            )

        for key in _REQUIRED_MODEL_KEYS:
            if key not in model_cfg:
                raise ConfigValidationError(
                    f"The 'model' section is missing required key {key!r}. "
                    f"Available keys: {list(model_cfg.keys())}"
                )

    # -- Train section ------------------------------------------------------

    @staticmethod
    def validate_train_config(config: Dict[str, Any]) -> None:
        """Validate the training configuration section.

        Checks:
            1. If a ``"train"`` section exists, it must be a dict.
            2. It contains required keys: ``batch_size``, ``epochs``.

        Parameters
        ----------
        config : dict
            The full configuration dictionary.

        Raises
        ------
        ConfigValidationError
            If the training section is present but structurally invalid.
        """
        ConfigSchema.validate_top_level(config)

        if "train" not in config:
            return  # Training config is optional

        train_cfg = config["train"]
        if not isinstance(train_cfg, dict):
            raise ConfigValidationError(
                f"The 'train' section must be a dict, got "
                f"{type(train_cfg).__name__!r}."
            )

        for key in _REQUIRED_TRAIN_KEYS:
            if key not in train_cfg:
                raise ConfigValidationError(
                    f"The 'train' section is missing required key {key!r}. "
                    f"Available keys: {list(train_cfg.keys())}"
                )

    # -- Optim section ------------------------------------------------------

    @staticmethod
    def validate_optim_config(config: Dict[str, Any]) -> None:
        """Validate the optimizer configuration section.

        Checks:
            1. If an ``"optim"`` section exists, it must be a dict.
            2. It contains required keys: ``name``, ``lr``.

        Parameters
        ----------
        config : dict
            The full configuration dictionary.

        Raises
        ------
        ConfigValidationError
            If the optim section is present but structurally invalid.
        """
        ConfigSchema.validate_top_level(config)

        if "optim" not in config:
            return  # Optim config is optional

        optim_cfg = config["optim"]
        if not isinstance(optim_cfg, dict):
            raise ConfigValidationError(
                f"The 'optim' section must be a dict, got "
                f"{type(optim_cfg).__name__!r}."
            )

        for key in _REQUIRED_OPTIM_KEYS:
            if key not in optim_cfg:
                raise ConfigValidationError(
                    f"The 'optim' section is missing required key {key!r}. "
                    f"Available keys: {list(optim_cfg.keys())}"
                )

    # -- Full validation ----------------------------------------------------

    @staticmethod
    def validate_all(config: Dict[str, Any]) -> None:
        """Run all validations on the configuration.

        Equivalent to calling ``validate_model_config``,
        ``validate_train_config``, and ``validate_optim_config``
        sequentially.

        Parameters
        ----------
        config : dict
            The full configuration dictionary.

        Raises
        ------
        ConfigValidationError
            If any validation fails.
        """
        ConfigSchema.validate_model_config(config)
        ConfigSchema.validate_train_config(config)
        ConfigSchema.validate_optim_config(config)
