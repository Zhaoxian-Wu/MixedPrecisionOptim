from torch.optim import SGD

from MixedPrecisionOptim.mpmixin import AnalogMpOptimizerMixin

class MpSGD(AnalogMpOptimizerMixin, SGD):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = SGD.step
