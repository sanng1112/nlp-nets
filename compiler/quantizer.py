"""
Quantizer — reduce model precision for faster inference and smaller footprints.

Supports:
    - **Dynamic quantization** (weights only, best for NLP models).
    - **Static quantization** (weights + activations, requires calibration).
    - **fp16 half-precision** conversion.

Usage:
    quantizer = Quantizer(model, opts)
    quantizer.quantize(mode="dynamic")
    quantizer.save("quantized_model.pt")
"""

from typing import Any, Dict, Optional

import torch

from utils import logger


class Quantizer:
    """
    Apply post-training quantization to a PyTorch model.

    Config keys (under ``compiler.quantization``):
        - ``mode``: ``"dynamic"`` (default), ``"static"``, or ``"fp16"``.
        - ``dtype``: Target quantization dtype (``"qint8"`` or ``"quint8"``).
        - ``calibration_method``: For static quantization (``"histogram"``, default).
        - ``backend``: Quantization backend (``"x86"``, ``"qnnpack"``, ``"fbgemm"``).

    Args:
        model: The PyTorch model to quantize.
        opts: Configuration dictionary (typically loaded from YAML).
    """

    def __init__(self, model: torch.nn.Module, opts: Dict[str, Any]) -> None:
        self.model = model
        self.opts = opts
        self.quantized_model: Optional[torch.nn.Module] = None

        quant_cfg = opts.get("compiler", {}).get("quantization", {})
        self.mode = quant_cfg.get("mode", "dynamic")
        self.dtype = self._parse_dtype(quant_cfg.get("dtype", "qint8"))
        self.backend = quant_cfg.get("backend", "x86")

    @staticmethod
    def _parse_dtype(dtype_str: str) -> torch.dtype:
        mapping = {
            "qint8": torch.qint8,
            "quint8": torch.quint8,
            "qint32": torch.qint32,
            "float16": torch.float16,
        }
        if dtype_str not in mapping:
            raise ValueError(
                f"Unknown quantization dtype: '{dtype_str}'. "
                f"Choose from {list(mapping.keys())}."
            )
        return mapping[dtype_str]

    def quantize(self, *args, **kwargs) -> None:
        """
        Quantize the model.

        Keyword Args:
            mode: ``"dynamic"``, ``"static"``, or ``"fp16"``.
            calibration_loader: DataLoader for static quantization calibration.
        """
        mode = kwargs.get("mode", self.mode)
        self.model.eval()

        if mode == "dynamic":
            self._dynamic_quantize()
        elif mode == "static":
            calibration_loader = kwargs.get("calibration_loader")
            self._static_quantize(calibration_loader)
        elif mode == "fp16":
            self._fp16_convert()
        else:
            raise ValueError(
                f"Unknown quantization mode: '{mode}'. "
                f"Use 'dynamic', 'static', or 'fp16'."
            )

    def _dynamic_quantize(self) -> None:
        """Apply dynamic quantization (weights only)."""
        logger.log("Quantizer: applying dynamic quantization...")
        self.quantized_model = torch.ao.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantise only linear layers
            dtype=self.dtype,
        )
        logger.log("Dynamic quantization complete.")

    def _static_quantize(self, calibration_loader: Any = None) -> None:
        """Apply static quantization (weights + activations)."""
        logger.log("Quantizer: applying static quantization...")
        torch.backends.quantized.engine = self.backend

        # Prepare model for quantization
        qconfig = torch.ao.quantization.get_default_qconfig(self.backend)
        self.model.qconfig = qconfig
        self.model.train()
        self.model = torch.ao.quantization.prepare(self.model)

        # Calibrate
        if calibration_loader is not None:
            logger.log("Quantizer: running calibration...")
            for batch in calibration_loader:
                if isinstance(batch, (list, tuple)):
                    self.model(*batch)
                elif isinstance(batch, dict):
                    self.model(**batch)
                else:
                    self.model(batch)
            logger.log("Calibration complete.")

        # Convert
        self.quantized_model = torch.ao.quantization.convert(self.model.eval())
        logger.log("Static quantization complete.")

    def _fp16_convert(self) -> None:
        """Convert model to half-precision (fp16)."""
        logger.log("Quantizer: converting to fp16...")
        self.quantized_model = self.model.half()
        logger.log("fp16 conversion complete.")

    def save(self, path: str) -> None:
        """
        Save the quantized model to disk.

        Args:
            path: Output file path (``.pt`` or ``.pth``).
        """
        if self.quantized_model is None:
            raise RuntimeError("No quantized model to save. Call quantize() first.")

        torch.save(self.quantized_model.state_dict(), path)
        logger.log(f"Quantized model saved to: {path}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mode={self.mode}, "
            f"quantized={self.quantized_model is not None})"
        )
