"""PyTorch SGD with MP storage and analog programming."""

from torch.optim import SGD
from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin


class MpSGD(MpOptimizerMixin, SGD):
    digital_optimizer_cls = SGD


__all__ = ["MpSGD"]
