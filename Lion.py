"""Digital Lion and its MP adapter."""

from __future__ import annotations
from typing import Any
import torch
from torch import Tensor

from MixedPrecisionOptim._digital_rule import _DigitalRuleOptimizer
from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin


class DigitalLion(_DigitalRuleOptimizer):
    """Lion (evolved sign momentum) on ordinary PyTorch parameters."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        self._validate_group(defaults)
        super().__init__(params, defaults)

    def _validate_group(self, group: dict[str, Any]) -> None:
        unsupported = {
            "eps", "amsgrad", "maximize", "foreach", "capturable",
            "differentiable", "fused", "shampoo_beta", "decoupled_weight_decay",
        }.intersection(group)
        if unsupported:
            raise TypeError(f"Lion does not support options: {sorted(unsupported)}")
        if not 0.0 <= group["lr"]:
            raise ValueError(f"Invalid learning rate: {group['lr']}")
        if not 0.0 <= group["weight_decay"]:
            raise ValueError(f"Invalid weight_decay value: {group['weight_decay']}")
        betas = group["betas"]
        if len(betas) != 2 or any(not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid beta parameters: {betas}")

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        momentum = state.setdefault("exp_avg", torch.zeros_like(param))
        beta1, beta2 = group["betas"]
        self._decay(param, group)
        update = momentum.lerp(grad, 1.0 - beta1).sign()
        param.add_(update, alpha=-float(group["lr"]))
        momentum.lerp_(grad, 1.0 - beta2)


class MpLion(MpOptimizerMixin, DigitalLion):
    digital_optimizer_cls = DigitalLion


__all__ = ["DigitalLion", "MpLion"]
