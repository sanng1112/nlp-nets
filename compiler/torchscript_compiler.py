"""
TorchScript compiler — trace or script a model with ``torch.jit``.

Supports two modes:
    - **trace**: Trace the model using example inputs (``torch.jit.trace``).
    - **script**: Script the model (``torch.jit.script``) — requires
      source-code compatibility.

Usage:
    compiler = TorchScriptCompiler(model, opts)
    compiler.compile(mode="trace", example_input=example_input)
    compiler.save("model.pt")
"""

from typing import Any, Dict, Optional

import torch

from compiler.base_compiler import BaseCompiler
from utils import logger


class TorchScriptCompiler(BaseCompiler):
    """
    Compile a model to TorchScript via tracing or scripting.

    Config keys (under ``compiler.torchscript``):
        - ``mode``: ``"trace"`` (default) or ``"script"``.
        - ``strict``: Whether to run in strict mode (default ``True``).
        - ``check_trace``: Whether to check traced outputs (default ``True``).
    """

    def __init__(self, model: torch.nn.Module, opts: Dict[str, Any]) -> None:
        super().__init__(model, opts)
        compiler_cfg = opts.get("compiler", {}).get("torchscript", {})
        self.mode = compiler_cfg.get("mode", "trace")
        self.strict = compiler_cfg.get("strict", True)
        self.check_trace = compiler_cfg.get("check_trace", True)

    def compile(self, *args, **kwargs) -> None:
        """
        Compile the model to TorchScript.

        Keyword Args:
            mode: ``"trace"`` or ``"script"``. Overrides config value.
            example_input: Example input tensor(s) for tracing.
                Required for ``mode="trace"``.
        """
        mode = kwargs.get("mode", self.mode)
        self.model.eval()

        if mode == "script":
            logger.log("TorchScriptCompiler: scripting model...")
            self.compiled_model = torch.jit.script(self.model, strict=self.strict)
        elif mode == "trace":
            example_input = kwargs.get("example_input")
            if example_input is None:
                raise ValueError(
                    "example_input is required for trace mode. "
                    "Pass it as a keyword argument to compile()."
                )
            logger.log("TorchScriptCompiler: tracing model...")
            self.compiled_model = torch.jit.trace(
                self.model,
                example_input,
                strict=self.strict,
                check_trace=self.check_trace,
            )
        else:
            raise ValueError(f"Unknown TorchScript mode: '{mode}'. Use 'trace' or 'script'.")

        logger.log(f"TorchScript compilation successful ({mode} mode).")

    def save(self, path: str) -> None:
        """
        Save the TorchScript model to disk.

        Args:
            path: Output ``.pt`` or ``.pth`` file path.
        """
        if self.compiled_model is None:
            raise RuntimeError("No compiled model to save. Call compile() first.")
        torch.jit.save(self.compiled_model, path)
        logger.log(f"TorchScript model saved to: {path}")

    def validate(self) -> bool:
        """
        Run a basic validation — ensure the compiled model is a
        ``ScriptModule`` and can run a forward pass.
        """
        if self.compiled_model is None:
            return False
        return isinstance(self.compiled_model, torch.jit.ScriptModule)
