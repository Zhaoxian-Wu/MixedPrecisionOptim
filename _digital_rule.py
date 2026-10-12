"""Shared lifecycle for project-specific digital rules, without algorithm equations."""

from __future__ import annotations
from typing import Any, Callable, Optional
import torch
from torch import Tensor
from torch.optim import Optimizer


class _DigitalRuleOptimizer(Optimizer):
    """Shared update loop; algorithms provide their own constructor and defaults.

    Subclasses call ``super().__init__(params, defaults)`` and implement group
    validation and parameter updates. Configuration is checked only when
    groups are registered or a checkpoint is loaded, never in the step loop.
    """

    def _validate_group(self, group: dict[str, Any]) -> None:
        raise NotImplementedError

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self._validate_group({**self.defaults, **param_group})
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for group in state_dict["param_groups"]:
            self._validate_group(group)
        super().load_state_dict(state_dict)

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    self._update_parameter(param, group)
        return loss

    @staticmethod
    def _decay(param: Tensor, group: dict[str, Any]) -> None:
        decay = float(group.get("weight_decay", 0.0))
        if decay:
            param.mul_(1.0 - float(group["lr"]) * decay)
