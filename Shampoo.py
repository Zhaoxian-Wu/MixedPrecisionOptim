"""Digital Shampoo and its MP adapter."""

from __future__ import annotations
from typing import Any
import torch
from torch import Tensor

from MixedPrecisionOptim._digital_rule import _DigitalRuleOptimizer
from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin


class DigitalShampoo(_DigitalRuleOptimizer):
    """Two-sided Shampoo for matrices, with a diagonal fallback otherwise."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        shampoo_beta: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        defaults = dict(lr=lr, shampoo_beta=shampoo_beta, eps=eps, weight_decay=weight_decay)
        self._validate_group(defaults)
        super().__init__(params, defaults)

    def _validate_group(self, group: dict[str, Any]) -> None:
        unsupported = {
            "betas", "amsgrad", "maximize", "foreach", "capturable",
            "differentiable", "fused", "decoupled_weight_decay",
        }.intersection(group)
        if unsupported:
            raise TypeError(f"Shampoo does not support options: {sorted(unsupported)}")
        if not 0.0 <= group["lr"]:
            raise ValueError(f"Invalid learning rate: {group['lr']}")
        if not 0.0 <= group["weight_decay"]:
            raise ValueError(f"Invalid weight_decay value: {group['weight_decay']}")
        if not 0.0 <= group["eps"]:
            raise ValueError(f"Invalid epsilon value: {group['eps']}")
        if not 0.0 <= group["shampoo_beta"] < 1.0:
            raise ValueError(f"Invalid Shampoo beta: {group['shampoo_beta']}")

    @staticmethod
    def _inverse_quarter_root(matrix: Tensor, epsilon: float) -> Tensor:
        values, vectors = torch.linalg.eigh(matrix)
        roots = values.clamp_min(epsilon).pow(-0.25)
        return (vectors * roots.unsqueeze(0)) @ vectors.transpose(0, 1)

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        beta = float(group["shampoo_beta"])
        eps = float(group["eps"])
        if grad.ndim >= 2:
            matrix = grad.reshape(grad.shape[0], -1)
            left = state.setdefault(
                "shampoo_left",
                torch.zeros(matrix.shape[0], matrix.shape[0], device=grad.device, dtype=grad.dtype),
            )
            right = state.setdefault(
                "shampoo_right",
                torch.zeros(matrix.shape[1], matrix.shape[1], device=grad.device, dtype=grad.dtype),
            )
            left.mul_(beta).addmm_(matrix, matrix.transpose(0, 1), beta=1.0, alpha=1.0 - beta)
            right.mul_(beta).addmm_(matrix.transpose(0, 1), matrix, beta=1.0, alpha=1.0 - beta)
            left_root = self._inverse_quarter_root(left, eps)
            right_root = self._inverse_quarter_root(right, eps)
            update = (left_root @ matrix @ right_root).reshape_as(grad)
        else:
            square_avg = state.setdefault("square_avg", torch.zeros_like(param))
            square_avg.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)
            update = grad / square_avg.sqrt().clamp_min(eps)
        self._decay(param, group)
        param.add_(update, alpha=-float(group["lr"]))


class MpShampoo(MpOptimizerMixin, DigitalShampoo):
    digital_optimizer_cls = DigitalShampoo


__all__ = ["DigitalShampoo", "MpShampoo"]
