"""
Compiler builder — constructs a compiler or quantizer from a config dict.

Supports the following backend types (under ``compiler.backend``):
    - ``"torchscript"`` → ``TorchScriptCompiler``
    - ``"onnx"``        → ``ONNXCompiler``
    - ``"torch_compile"`` → ``TorchCompileCompiler``
    - ``"quantize"``     → ``Quantizer`` (quantization-only pipeline)
"""

from typing import Any, Dict, Optional

import torch

from compiler.onnx_compiler import ONNXCompiler
from compiler.quantizer import Quantizer
from compiler.torch_compile_compiler import TorchCompileCompiler
from compiler.torchscript_compiler import TorchScriptCompiler
from utils import logger

# Registry mapping backend names to compiler classes
_COMPILER_REGISTRY = {
    "torchscript": TorchScriptCompiler,
    "onnx": ONNXCompiler,
    "torch_compile": TorchCompileCompiler,
    "quantize": Quantizer,
}


def build_compiler(
    opts: Dict[str, Any],
    model: torch.nn.Module,
) -> Any:
    """
    Build a compiler or quantizer instance from configuration.

    Args:
        opts: Configuration dictionary (typically loaded from YAML).
            Expects ``opts["compiler"]["backend"]`` to select the backend.
        model: The PyTorch model to compile or quantize.

    Returns:
        An initialised compiler or quantizer instance.

    Raises:
        ValueError: If the backend is unknown or not specified.
    """
    compiler_cfg = opts.get("compiler", {})
    backend = compiler_cfg.get("backend", "").strip().lower()

    if not backend:
        raise ValueError(
            "Compiler backend must be specified in config under 'compiler.backend'. "
            f"Available backends: {list(_COMPILER_REGISTRY.keys())}"
        )

    if backend not in _COMPILER_REGISTRY:
        raise ValueError(
            f"Unknown compiler backend '{backend}'. "
            f"Available backends: {list(_COMPILER_REGISTRY.keys())}"
        )

    compiler_cls = _COMPILER_REGISTRY[backend]
    compiler = compiler_cls(model, opts)
    logger.log(f"Compiler built: {type(compiler).__name__} (backend='{backend}')")
    return compiler
