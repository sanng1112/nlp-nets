"""
Tests for optimizer and scheduler builders.
"""

import torch
import torch.nn as nn
import pytest

from optim.optimizer_builder import build_optimizer
from optim.scheduler_builder import build_scheduler


class _DummyModel(nn.Module):
    """Minimal model for optimizer tests."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestOptimizerBuilder:
    """Test optimizer construction."""

    def test_build_adamw(self):
        model = _DummyModel()
        opts = {
            "optim": {
                "name": "adamw",
                "lr": 3e-4,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
            }
        }
        optim = build_optimizer(opts, model)
        assert isinstance(optim, torch.optim.AdamW)
        assert optim.param_groups[0]["lr"] == 3e-4

    def test_build_sgd(self):
        model = _DummyModel()
        opts = {
            "optim": {
                "name": "sgd",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            }
        }
        optim = build_optimizer(opts, model)
        assert isinstance(optim, torch.optim.SGD)
        assert optim.param_groups[0]["lr"] == 0.1

    def test_unknown_optimizer(self):
        model = _DummyModel()
        opts = {"optim": {"name": "unknown_optim"}}
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            build_optimizer(opts, model)


class TestSchedulerBuilder:
    """Test learning rate scheduler construction."""

    def test_build_cosine_scheduler(self):
        model = _DummyModel()
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
        opts = {
            "optim": {
                "scheduler": {
                    "name": "cosine",
                    "t_max": 10,
                    "min_lr": 1e-6,
                }
            }
        }
        scheduler = build_scheduler(opts, optim)
        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_no_scheduler(self):
        model = _DummyModel()
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = build_scheduler({}, optim)
        assert scheduler is None
