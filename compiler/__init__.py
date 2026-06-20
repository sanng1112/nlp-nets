"""
nlp-nets compiler: Model optimization and deployment utilities.

Provides tools to compile, optimize, quantize, and export NLP models
for production inference. Supports three compilation backends:

    - **TorchScript**: Trace or script models via ``torch.jit``.
    - **ONNX**: Export to ONNX format for cross-platform inference.
    - **torch.compile**: Use PyTorch 2.0+ graph compilation (default).

Usage:
    from compiler import build_compiler

    compiler = build_compiler(opts, model)
    compiler.compile()
    compiler.save("output_path")
"""

from compiler.base_compiler import BaseCompiler
from compiler.torchscript_compiler import TorchScriptCompiler
from compiler.onnx_compiler import ONNXCompiler
from compiler.torch_compile_compiler import TorchCompileCompiler
from compiler.quantizer import Quantizer
from compiler.builder import build_compiler

__all__ = [
    "BaseCompiler",
    "TorchScriptCompiler",
    "ONNXCompiler",
    "TorchCompileCompiler",
    "Quantizer",
    "build_compiler",
]
