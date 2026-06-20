"""
ONNX compiler — export a model to the ONNX format for cross-platform inference.

Supports:
    - Dynamic axes for variable-length sequences.
    - Mixed-precision (fp16) export.
    - Operator-set selection.

Usage:
    compiler = ONNXCompiler(model, opts)
    compiler.compile(example_input=example_input)
    compiler.save("model.onnx")
"""

import io
from typing import Any, Dict, Optional, Sequence

import torch

from compiler.base_compiler import BaseCompiler
from utils import logger


class ONNXCompiler(BaseCompiler):
    """
    Export a PyTorch model to ONNX format.

    Config keys (under ``compiler.onnx``):
        - ``opset_version``: ONNX opset version (default ``17``).
        - ``dynamic_axes``: Dict of dynamic axes (default for batch & seq).
        - ``export_fp16``: Whether to export in fp16 (default ``False``).
        - ``input_names``: List of input tensor names (default ``["input_ids"]``).
        - ``output_names``: List of output tensor names (default ``["logits"]``).
    """

    def __init__(self, model: torch.nn.Module, opts: Dict[str, Any]) -> None:
        super().__init__(model, opts)
        compiler_cfg = opts.get("compiler", {}).get("onnx", {})
        self.opset_version = compiler_cfg.get("opset_version", 17)
        self.export_fp16 = compiler_cfg.get("export_fp16", False)
        self.input_names = compiler_cfg.get("input_names", ["input_ids"])
        self.output_names = compiler_cfg.get("output_names", ["logits"])
        self.dynamic_axes = compiler_cfg.get(
            "dynamic_axes",
            {
                0: "batch_size",
                1: "seq_length",
            },
        )

    @staticmethod
    def _check_dependencies() -> None:
        """Ensure optional ONNX-related packages are available."""
        try:
            import onnx  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'onnx' package is required for ONNX export. "
                "Install it with: pip install onnx"
            )

    def _export_to_bytes(self, example_input, dynamic_axes) -> bytes:
        """Run torch.onnx.export and return serialized ONNX proto as bytes."""
        buf = io.BytesIO()
        torch.onnx.export(
            self.model,
            example_input,
            buf,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
        )
        return buf.getvalue()

    def compile(self, *args, **kwargs) -> None:
        """
        Export the model to ONNX format.

        Keyword Args:
            example_input: Example input tensor(s) for tracing.
                Required. Can be a single tensor or tuple of tensors.
        """
        self._check_dependencies()

        example_input = kwargs.get("example_input")
        if example_input is None:
            raise ValueError(
                "example_input is required for ONNX export. "
                "Pass it as a keyword argument to compile()."
            )

        self.model.eval()
        device = next(self.model.parameters()).device

        # Move example input to the same device as the model
        if isinstance(example_input, torch.Tensor):
            example_input = example_input.to(device)
        elif isinstance(example_input, (tuple, list)):
            example_input = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t
                for t in example_input
            )
        elif isinstance(example_input, dict):
            example_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in example_input.items()
            }

        # Determine dynamic axes for each input
        if isinstance(example_input, dict):
            dynamic_axes = {}
            for name in self.input_names:
                if name in example_input:
                    dynamic_axes[name] = self.dynamic_axes
        else:
            dynamic_axes = {name: self.dynamic_axes for name in self.input_names}

        logger.log("ONNXCompiler: exporting model to ONNX...")
        self.compiled_model = self._export_to_bytes(example_input, dynamic_axes)
        logger.log("ONNX export successful.")

        # Convert to fp16 if requested
        if self.export_fp16:
            logger.log("ONNXCompiler: converting to fp16...")
            try:
                import onnx
                from onnxconverter_common import float16

                onnx_model = onnx.load_model_from_string(self.compiled_model)
                onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
                buf = io.BytesIO()
                onnx.save(onnx_model_fp16, buf)
                self.compiled_model = buf.getvalue()
                logger.log("ONNXCompiler: fp16 conversion complete.")
            except ImportError:
                logger.warning(
                    "onnx or onnxconverter-common not installed. "
                    "Skipping fp16 conversion."
                )

    def save(self, path: str) -> None:
        """
        Save the ONNX model to disk.

        Args:
            path: Output ``.onnx`` file path.
        """
        if self.compiled_model is None:
            raise RuntimeError("No compiled model to save. Call compile() first.")

        with open(path, "wb") as f:
            f.write(self.compiled_model)
        logger.log(f"ONNX model saved to: {path}")
