"""PyTorch AdamW with MP storage and analog programming."""

from torch.optim import AdamW
from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin


class MpAdamW(MpOptimizerMixin, AdamW):
    digital_optimizer_cls = AdamW


__all__ = ["MpAdamW"]
