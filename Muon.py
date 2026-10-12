"""Muon adapters: PyTorch Muon matrices and PyTorch AdamW auxiliaries."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin


class NativeMatrixWithAdamW(Optimizer):
    """Native Muon for matrices, native AdamW for auxiliary parameters."""

    matrix_lr_adjustment = "original"

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        *,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ) -> None:
        if not hasattr(torch.optim, "Muon"):
            raise ImportError("Muon/Moonlight require a PyTorch build providing torch.optim.Muon")
        matrix_optimizer = torch.optim.Muon(
            [{"params": []}], lr=lr, weight_decay=weight_decay,
            momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            eps=ns_eps, ns_coefficients=ns_coefficients,
            adjust_lr_fn=self.matrix_lr_adjustment,
        )
        aux_optimizer = torch.optim.AdamW(
            [{"params": []}], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, maximize=maximize, foreach=foreach,
            capturable=capturable, differentiable=differentiable, fused=fused,
        )
        defaults = dict(
            aux_optimizer.defaults,
            momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            ns_eps=ns_eps, ns_coefficients=ns_coefficients,
        )
        super().__init__(params, defaults)
        self._matrix_optimizer = matrix_optimizer
        self._aux_optimizer = aux_optimizer

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        matrix_groups, aux_groups, views = [], [], []
        try:
            for group in self.param_groups:
                matrix_params, aux_params = [], []
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if not (group.get("use_muon", param.ndim >= 2) and param.ndim >= 2):
                        aux_params.append(param)
                        continue
                    if param.ndim == 2 and not group.get("maximize", False):
                        matrix_params.append(param)
                        continue
                    matrix = nn.Parameter(param.detach().reshape(param.shape[0], -1))
                    gradient = param.grad.reshape_as(matrix)
                    matrix.grad = -gradient if group.get("maximize", False) else gradient
                    state = dict(self.state[param])
                    if "momentum_buffer" in state:
                        state["momentum_buffer"] = state["momentum_buffer"].reshape_as(matrix)
                    self.state[matrix] = state
                    views.append((param, matrix))
                    matrix_params.append(matrix)
                if matrix_params:
                    if group.get("differentiable") or group.get("capturable"):
                        raise NotImplementedError(
                            "Native Muon does not support differentiable/capturable steps"
                        )
                    matrix_groups.append(
                        dict(
                            params=matrix_params,
                            lr=group["lr"],
                            weight_decay=group.get("weight_decay", 0),
                            momentum=group.get("momentum", self.defaults["momentum"]),
                            nesterov=group.get("nesterov", self.defaults["nesterov"]),
                            ns_coefficients=group.get("ns_coefficients", (3.4445, -4.775, 2.0315)),
                            ns_steps=group.get("ns_steps", self.defaults["ns_steps"]),
                            eps=group.get("ns_eps", self.defaults["ns_eps"]),
                            adjust_lr_fn=self.matrix_lr_adjustment,
                        )
                    )
                if aux_params:
                    aux_groups.append(
                        {
                            **{key: group.get(key, value)
                               for key, value in self._aux_optimizer.defaults.items()},
                            "params": aux_params,
                        }
                    )
            for delegate, groups in (
                (self._matrix_optimizer, matrix_groups),
                (self._aux_optimizer, aux_groups),
            ):
                delegate.param_groups = groups
                delegate.state = self.state
                native_step = type(delegate).step
                if getattr(native_step, "hooked", False):
                    native_step = native_step.__wrapped__
                native_step(delegate)
        finally:
            for param, matrix in views:
                if matrix.data_ptr() != param.data_ptr():
                    # Non-contiguous convolution weights need a temporary
                    # contiguous matrix; write it back before releasing it.
                    param.copy_(matrix.reshape_as(param))
                state = self.state.pop(matrix)
                if "momentum_buffer" in state:
                    state["momentum_buffer"] = state["momentum_buffer"].reshape_as(param)
                self.state[param] = state
                matrix.grad = None
            self._matrix_optimizer.param_groups = []
            self._aux_optimizer.param_groups = []
            self._matrix_optimizer.state = {}
            self._aux_optimizer.state = {}
        return loss


class DigitalMuon(NativeMatrixWithAdamW):
    """Muon with original aspect-ratio scaling; flatten convolutional weights."""

    matrix_lr_adjustment = "original"


class MpMuon(MpOptimizerMixin, DigitalMuon):
    digital_optimizer_cls = DigitalMuon


class MuonWithAuxAdam(DigitalMuon):
    """Compatibility constructor with explicitly marked Muon/AdamW groups."""

    def __init__(self, param_groups):
        groups = []
        for original in param_groups:
            group = dict(original)
            if "use_muon" not in group:
                raise ValueError("MuonWithAuxAdam requires use_muon in each parameter group")
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda param: param.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
            group.setdefault("weight_decay", 0)
            groups.append(group)
        super().__init__(groups, weight_decay=0)


class Muon(DigitalMuon):
    """Matrix-only compatibility constructor using native Muon updates."""

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        super().__init__(
            [dict(params=list(params), use_muon=True)],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )


SingleDeviceMuon = Muon
SingleDeviceMuonWithAuxAdam = MuonWithAuxAdam


@torch.no_grad()
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, scale_by_aspect_ratio=True):
    """Compatibility helper delegated to PyTorch's Muon functional API."""
    from torch.optim._muon import muon

    matrix = grad.reshape(grad.shape[0], -1)
    result = torch.zeros_like(matrix)
    ratio = max(1, matrix.shape[0] / matrix.shape[1]) ** 0.5
    muon(
        [result],
        [matrix],
        [momentum.reshape_as(matrix)],
        lr=1.0 if scale_by_aspect_ratio else 1.0 / ratio,
        weight_decay=0,
        momentum=beta,
        nesterov=nesterov,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        ns_steps=ns_steps,
        eps=1e-7,
        adjust_lr_fn="original",
        has_complex=False,
    )
    return -result


__all__ = [
    "DigitalMuon",
    "MpMuon",
    "Muon",
    "MuonWithAuxAdam",
    "SingleDeviceMuon",
    "SingleDeviceMuonWithAuxAdam",
    "muon_update",
]
