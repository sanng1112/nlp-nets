"""
Torch compile compiler — use ``torch.compile`` (PyTorch 2.0+) for
graph-level optimization.

Supports multiple backends:
    - ``"default"``: The default inductor-based backend.
    - ``"inductor"``: Explicit inductor backend.
    - ``"cudagraphs"``: CUDA graph capture.
    - ``"fx2trt"``: NVIDIA TensorRT via fx (if installed).

Usage:
    compiler = TorchCompileCompiler(model, opts)
    compiler.compile()
    compiler.save("compiled_model.pt")
"""

from typing import Any, Dict, Optional

import torch

from compiler.base_compiler import BaseCompiler
from utils import logger


class TorchCompileCompiler(BaseCompiler):
    """
    Optimize a model via ``torch.compile``.

    Config keys (under ``compiler.torch_compile``):
        - ``backend``: Compilation backend (default ``"inductor"``).
        - ``mode``: Compilation mode (default ``None``, uses torch default).
            Options: ``"default"``, ``"reduce-overhead"``, ``"max-autotune"``.
        - ``fullgraph``: Require a single full graph (default ``False``).
        - ``dynamic``: Enable dynamic shape support (default ``False``).

    Reference: https://pytorch.org/docs/stable/generated/torch.compile.html
    """

    def __init__(self, model: torch.nn.Module, opts: Dict[str, Any]) -> None:
        super().__init__(model, opts)
        compiler_cfg = opts.get("compiler", {}).get("torch_compile", {})
        self.backend = compiler_cfg.get("backend", "inductor")
        self.mode = compiler_cfg.get("mode", None)
        self.fullgraph = compiler_cfg.get("fullgraph", False)
        self.dynamic = compiler_cfg.get("dynamic", False)

    def compile(self, *args, **kwargs) -> None:
        """
        Compile the model using ``torch.compile``.

        ``self.compiled_model`` will be the optimized model callable.
        """
        logger.log(
            f"TorchCompileCompiler: compiling model with backend='{self.backend}', "
            f"mode='{self.mode}', fullgraph={self.fullgraph}, dynamic={self.dynamic}"
        )

        self.compiled_model = torch.compile(
            self.model,
            backend=self.backend,
            mode=self.mode,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )

        logger.log("torch.compile compilation successful.")

    def save(self, path: str) -> None:
        """
        Save the original model's state_dict (``torch.compile`` does not
        produce a serialisable artifact directly).

        For deployment with ``torch.compile``, save the state_dict and
        re-apply ``torch.compile`` at load time.

        Args:
            path: Output ``.pt`` or ``.pth`` file path.
        """
        if self.compiled_model is None:
            raise RuntimeError("No compiled model to save. Call compile() first.")

        # Save the state dict of the original (unwrapped) model
        torch.save(self.model.state_dict(), path)
        logger.log(
            f"Model state_dict saved to: {path} "
            f"(re-apply torch.compile after loading)"
        )
