import math
import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model weights.

    Maintains a shadow copy of the model parameters updated via exponential
    moving average. Provides smoother and more robust validation metrics.

    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        """
        Args:
            model: The model to shadow-copy.
            decay: EMA decay factor (higher = slower adaptation).
            tau: Temperature that controls the ramp-up of decay.
            updates: Initial update counter.
        """
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = updates

    def update(self, model: nn.Module):
        """Update EMA weights with the current model weights."""
        self.updates += 1
        # Ramp up decay in early steps
        d = self.decay * (1 - math.exp(-self.updates / self.tau))

        # Handle DDP/DataParallel wrappers
        msd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        with torch.no_grad():
            for name, param in self.ema.state_dict().items():
                if param.dtype.is_floating_point:
                    param.copy_(param * d + (1.0 - d) * msd[name].detach())

    def update_attr(self, model: nn.Module, include=(), exclude=("process_group", "reducer")):
        """Copy non-tensor attributes from the training model to the EMA model."""
        import inspect

        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k in exclude:
                continue
            if not k.startswith("_") and not inspect.ismethod(v):
                setattr(self.ema, k, v)
