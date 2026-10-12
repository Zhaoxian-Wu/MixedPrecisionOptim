from torch.optim import SGD

from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin
from MixedPrecisionOptim.agmpmixin import AnalogGradMpOptimizerMixin
from aihwkit.optim.context import AnalogContext


class MpSignSGD(MpOptimizerMixin, SGD):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """

    digital_optimizer_cls = SGD

    def prepare_grad(self) -> None:
        super().prepare_grad()
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    if param.grad is not None:
                        param.grad.sign_()


class AGMpSignSGD(AnalogGradMpOptimizerMixin, SGD):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """

    digital_optimizer_cls = SGD

    def prepare_grad(self) -> None:
        super().prepare_grad()
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    if param.grad is not None:
                        param.grad.sign_()
