from torch.optim import AdamW

from MixedPrecisionOptim.mpmixin import (
    AnalogGradMpMwOptimizerMixin,
    AnalogGradMpOptimizerMixin,
    AnalogMpOptimizerMixin,
    RandomDirectionMpOptimizerMixin,
    ZerothOrderOptimizerMixin,
)
from MixedPrecisionOptim.mpmixinexp import AnalogGradMpDRUOptimizerMixin

class MpAdamW(AnalogMpOptimizerMixin, AdamW):
    """Mixed-precision AdamW for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.

    Note:
        The decoupled weight decay path is not available yet for ``MpAdamW``.
        The ``weight_decay`` argument is therefore left for future work.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step

class AGMpAdamW(AnalogGradMpOptimizerMixin, AdamW):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step


class RandomMpAdamW(AnalogGradMpOptimizerMixin, AdamW):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step
        
    def prepare_grad(self):
        return super().prepare_grad()



class AGMpDRUAdamW(AnalogGradMpDRUOptimizerMixin, AdamW):
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step

class AGMpMwAdamW(AnalogGradMpMwOptimizerMixin, AdamW):
    """Mixin for analog optimizers with half-order update.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step
      
class RandomDirectionMpAdamW(RandomDirectionMpOptimizerMixin, AdamW):
    """Mixin for analog optimizers with half-order update.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step


class ZerothOrderMpAdamW(ZerothOrderOptimizerMixin, AdamW):
    """Zero-Order optimizer with mixed precision for analog hardware.

    This class contains the methods needed for enabling zero-order optimization
    on analog hardware in an existing ``Optimizer``. It is designed to be used as
    a mixin in conjunction with an ``AnalogOptimizer`` or torch ``Optimizer``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step
