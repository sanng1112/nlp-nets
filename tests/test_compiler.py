"""
Unit tests for nlp-nets compiler module.
"""

import io
from pathlib import Path
from typing import Dict, Any

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    """A tiny MLP that can be compiled by all backends."""
    class SimpleMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(32, 16)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    return SimpleMLP()


@pytest.fixture
def compiler_opts() -> Dict[str, Any]:
    return {
        "compiler": {
            "backend": "torchscript",
            "torchscript": {
                "mode": "trace",
                "strict": True,
                "check_trace": True,
            },
            "onnx": {
                "opset_version": 17,
                "input_names": ["input_ids"],
                "output_names": ["logits"],
            },
            "torch_compile": {
                "backend": "inductor",
                "mode": None,
                "fullgraph": False,
                "dynamic": False,
            },
            "quantization": {
                "mode": "dynamic",
                "dtype": "qint8",
                "backend": "x86",
            },
        }
    }


@pytest.fixture
def example_input():
    return torch.randn(2, 32)


# ---------------------------------------------------------------------------
# BaseCompiler Tests
# ---------------------------------------------------------------------------

class TestBaseCompiler:
    def test_abstract_instantiation(self):
        """BaseCompiler cannot be instantiated directly."""
        from compiler.base_compiler import BaseCompiler

        with pytest.raises(TypeError):
            BaseCompiler(None, {})  # type: ignore[abstract]

    def test_base_repr(self, simple_model, compiler_opts):
        """Compiler classes have a readable repr."""
        from compiler.torchscript_compiler import TorchScriptCompiler

        c = TorchScriptCompiler(simple_model, compiler_opts)
        rep = repr(c)
        assert "TorchScriptCompiler" in rep
        assert "compiled=False" in rep


# ---------------------------------------------------------------------------
# TorchScriptCompiler Tests
# ---------------------------------------------------------------------------

class TestTorchScriptCompiler:
    def test_trace_compile(self, simple_model, compiler_opts, example_input):
        from compiler.torchscript_compiler import TorchScriptCompiler

        compiler = TorchScriptCompiler(simple_model, compiler_opts)
        compiler.compile(mode="trace", example_input=example_input)
        assert compiler.compiled_model is not None

        # Forward pass works
        with torch.no_grad():
            out = compiler.compiled_model(example_input)
        assert out.shape == (2, 8)

    def test_validate(self, simple_model, compiler_opts, example_input):
        from compiler.torchscript_compiler import TorchScriptCompiler

        compiler = TorchScriptCompiler(simple_model, compiler_opts)
        compiler.compile(mode="trace", example_input=example_input)
        assert compiler.validate() is True

    def test_save_and_load(self, simple_model, compiler_opts, example_input, tmp_path):
        from compiler.torchscript_compiler import TorchScriptCompiler

        compiler = TorchScriptCompiler(simple_model, compiler_opts)
        compiler.compile(mode="trace", example_input=example_input)

        save_path = str(tmp_path / "model.pt")
        compiler.save(save_path)

        # Load back
        loaded = torch.jit.load(save_path)
        with torch.no_grad():
            out = loaded(example_input)
        assert out.shape == (2, 8)

    def test_compile_no_example_raises(self, simple_model, compiler_opts):
        from compiler.torchscript_compiler import TorchScriptCompiler

        compiler = TorchScriptCompiler(simple_model, compiler_opts)
        with pytest.raises(ValueError, match="example_input"):
            compiler.compile(mode="trace")

    def test_save_before_compile_raises(self, simple_model, compiler_opts):
        from compiler.torchscript_compiler import TorchScriptCompiler

        compiler = TorchScriptCompiler(simple_model, compiler_opts)
        with pytest.raises(RuntimeError, match="compile"):
            compiler.save("dummy.pt")

    def test_trace_mode_from_config(self, simple_model, compiler_opts, example_input):
        """Default mode 'trace' is read from config."""
        from compiler.torchscript_compiler import TorchScriptCompiler

        compiler = TorchScriptCompiler(simple_model, compiler_opts)
        compiler.compile(example_input=example_input)
        assert isinstance(compiler.compiled_model, torch.jit.ScriptModule)


# ---------------------------------------------------------------------------
# ONNXCompiler Tests
# ---------------------------------------------------------------------------

class TestONNXCompiler:
    def test_export(self, simple_model, compiler_opts, example_input):
        pytest.importorskip("onnx", reason="onnx package not installed")
        from compiler.onnx_compiler import ONNXCompiler

        compiler = ONNXCompiler(simple_model, compiler_opts)
        compiler.compile(example_input=example_input)
        assert compiler.compiled_model is not None

    def test_save(self, simple_model, compiler_opts, example_input, tmp_path):
        pytest.importorskip("onnx", reason="onnx package not installed")
        from compiler.onnx_compiler import ONNXCompiler

        compiler = ONNXCompiler(simple_model, compiler_opts)
        compiler.compile(example_input=example_input)

        save_path = str(tmp_path / "model.onnx")
        compiler.save(save_path)
        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0

    def test_compile_no_example_raises(self, simple_model, compiler_opts):
        pytest.importorskip("onnx", reason="onnx package not installed")
        from compiler.onnx_compiler import ONNXCompiler

        compiler = ONNXCompiler(simple_model, compiler_opts)
        with pytest.raises(ValueError, match="example_input"):
            compiler.compile()

    def test_save_before_compile_raises(self, simple_model, compiler_opts):
        pytest.importorskip("onnx", reason="onnx package not installed")
        from compiler.onnx_compiler import ONNXCompiler

        compiler = ONNXCompiler(simple_model, compiler_opts)
        with pytest.raises(RuntimeError, match="compile"):
            compiler.save("dummy.onnx")


# ---------------------------------------------------------------------------
# TorchCompileCompiler Tests
# ---------------------------------------------------------------------------

class TestTorchCompileCompiler:
    def test_compile(self, simple_model, compiler_opts):
        from compiler.torch_compile_compiler import TorchCompileCompiler

        compiler = TorchCompileCompiler(simple_model, compiler_opts)
        compiler.compile()
        assert compiler.compiled_model is not None

    def test_save(self, simple_model, compiler_opts, tmp_path):
        from compiler.torch_compile_compiler import TorchCompileCompiler

        compiler = TorchCompileCompiler(simple_model, compiler_opts)
        compiler.compile()

        save_path = str(tmp_path / "compiled.pt")
        compiler.save(save_path)
        assert Path(save_path).exists()

        # Ensure it's a valid state_dict
        state = torch.load(save_path)
        assert isinstance(state, dict)
        assert any("fc1" in k for k in state)

    def test_save_before_compile_raises(self, simple_model, compiler_opts):
        from compiler.torch_compile_compiler import TorchCompileCompiler

        compiler = TorchCompileCompiler(simple_model, compiler_opts)
        with pytest.raises(RuntimeError, match="compile"):
            compiler.save("dummy.pt")


# ---------------------------------------------------------------------------
# Quantizer Tests
# ---------------------------------------------------------------------------

class TestQuantizer:
    def test_dynamic_quantize(self, simple_model, compiler_opts):
        from compiler.quantizer import Quantizer

        quantizer = Quantizer(simple_model, compiler_opts)
        quantizer.quantize(mode="dynamic")
        assert quantizer.quantized_model is not None

        # Forward pass still works
        x = torch.randn(2, 32)
        with torch.no_grad():
            out = quantizer.quantized_model(x)
        assert out.shape == (2, 8)

    def test_fp16_convert(self, simple_model, compiler_opts):
        from compiler.quantizer import Quantizer

        quantizer = Quantizer(simple_model, compiler_opts)
        quantizer.quantize(mode="fp16")
        assert quantizer.quantized_model is not None

        x = torch.randn(2, 32).half()
        with torch.no_grad():
            out = quantizer.quantized_model(x)
        assert out.shape == (2, 8)

    def test_save(self, simple_model, compiler_opts, tmp_path):
        from compiler.quantizer import Quantizer

        quantizer = Quantizer(simple_model, compiler_opts)
        quantizer.quantize(mode="dynamic")

        save_path = str(tmp_path / "quantized.pt")
        quantizer.save(save_path)
        assert Path(save_path).exists()

    def test_unknown_mode_raises(self, simple_model, compiler_opts):
        from compiler.quantizer import Quantizer

        quantizer = Quantizer(simple_model, compiler_opts)
        with pytest.raises(ValueError, match="quant"):
            quantizer.quantize(mode="unknown_mode")

    def test_save_before_quantize_raises(self, simple_model, compiler_opts):
        from compiler.quantizer import Quantizer

        quantizer = Quantizer(simple_model, compiler_opts)
        with pytest.raises(RuntimeError, match="quantize"):
            quantizer.save("dummy.pt")

    def test_repr(self, simple_model, compiler_opts):
        from compiler.quantizer import Quantizer

        quantizer = Quantizer(simple_model, compiler_opts)
        rep = repr(quantizer)
        assert "Quantizer" in rep
        assert "quantized=False" in rep


# ---------------------------------------------------------------------------
# Builder Tests
# ---------------------------------------------------------------------------

class TestBuildCompiler:
    def test_build_torchscript(self, simple_model, compiler_opts):
        from compiler.builder import build_compiler

        compiler = build_compiler(compiler_opts, simple_model)
        from compiler.torchscript_compiler import TorchScriptCompiler
        assert isinstance(compiler, TorchScriptCompiler)

    def test_build_onnx(self, simple_model, compiler_opts):
        from compiler.builder import build_compiler

        onnx_opts = {"compiler": {"backend": "onnx"}}
        compiler = build_compiler(onnx_opts, simple_model)
        from compiler.onnx_compiler import ONNXCompiler
        assert isinstance(compiler, ONNXCompiler)

    def test_build_torch_compile(self, simple_model, compiler_opts):
        from compiler.builder import build_compiler

        tc_opts = {"compiler": {"backend": "torch_compile"}}
        compiler = build_compiler(tc_opts, simple_model)
        from compiler.torch_compile_compiler import TorchCompileCompiler
        assert isinstance(compiler, TorchCompileCompiler)

    def test_build_quantize(self, simple_model, compiler_opts):
        from compiler.builder import build_compiler

        q_opts = {"compiler": {"backend": "quantize"}}
        compiler = build_compiler(q_opts, simple_model)
        from compiler.quantizer import Quantizer
        assert isinstance(compiler, Quantizer)

    def test_build_unknown_backend_raises(self, simple_model):
        from compiler.builder import build_compiler

        opts = {"compiler": {"backend": "nonexistent"}}
        with pytest.raises(ValueError, match="Unknown compiler backend"):
            build_compiler(opts, simple_model)

    def test_build_no_backend_raises(self, simple_model):
        from compiler.builder import build_compiler

        with pytest.raises(ValueError, match="must be specified"):
            build_compiler({}, simple_model)
