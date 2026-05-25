from torch.optim import SGD

from MixedPrecisionOptim.mpmixin import AnalogGradMpOptimizerMixin, AnalogMpOptimizerMixin
from aihwkit.optim.context import AnalogContext

class MpSignSGD(AnalogMpOptimizerMixin, SGD):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = SGD.step

    def prepare_grad(self) -> None:
        super().prepare_grad()
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.grad.sign_()

class AGMpSignSGD(AnalogGradMpOptimizerMixin, SGD):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = SGD.step

    def prepare_grad(self) -> None:
        super().prepare_grad()
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.grad.sign_()
                    