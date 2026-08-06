from torch.optim import AdamW

from MixedPrecisionOptim.mpmixin import AnalogMpOptimizerMixin


class MpAdamW(AnalogMpOptimizerMixin, AdamW):
    """Mixed-precision AdamW for analog optimizers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step
