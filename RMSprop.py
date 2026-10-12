"""Use the standard PyTorch RMSprop update, including momentum and centering."""

from torch.optim import RMSprop
from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin

DigitalRMSprop = RMSprop


class MpRMSprop(MpOptimizerMixin, RMSprop):
    digital_optimizer_cls = RMSprop

__all__ = ["DigitalRMSprop", "MpRMSprop"]
