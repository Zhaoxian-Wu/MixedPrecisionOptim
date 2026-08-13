"""Open-loop mixed-precision AdamW updates in logical-weight units.

This optimizer is intentionally independent of :class:`MpAdamW`.  The latter
is the historical baseline and interprets the ``AnalogContext`` buffer in the
tile's weight units.  Here the array stores ``hat W`` while the model uses
``W = s * hat W``.  Gradients, Adam moments, learning rates, and the transfer
residual are consequently all expressed in logical ``W`` units.

Array programming is open loop, matching MpAdamW: after requesting physical
pulses, the BUFFER is reduced by the corresponding *nominal* logical update.
The optimizer never reads the conductance matrix to settle a write.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import no_grad
from torch.optim import AdamW

from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs.devices import FloatingPointDevice

from MixedPrecisionOptim.mpmixin import AnalogMpOptimizerMixin


# Version 1 used closed-loop readback settlement and a logical shadow. It is
# intentionally incompatible with the open-loop state introduced here.
LOGICAL_MP_STATE_VERSION = 2

_VERSION_KEY = "logical_mp_version"
_SCALE_KEY = "logical_weight_scale"
_RESIDUAL_KEY = "logical_residual"
_TRANSFER_STEPS_KEY = "logical_transfer_steps"

_REQUIRED_LOGICAL_STATE_KEYS = {
    _VERSION_KEY,
    _SCALE_KEY,
    _RESIDUAL_KEY,
    _TRANSFER_STEPS_KEY,
}


class LogicalMpAdamW(AnalogMpOptimizerMixin, AdamW):
    """AdamW in logical units with an MpAdamW-style open-loop transfer.

    A wrapped analog layer binds a positive scalar ``logical_weight_scale`` to
    each :class:`~aihwkit.optim.context.AnalogContext`.  If the physical array
    stores ``hat W``, this scalar defines the logical matrix ``W = s * hat W``.

    The ``AnalogContext`` BUFFER is the pending logical-weight increment. A
    regular PyTorch AdamW step therefore updates this BUFFER using logical
    gradients and standard logical-space ``lr``, ``betas``, ``eps``, and Adam
    moments. For a pulsed device, complete logical quanta ``s * dw_min`` are
    sent as physical ``dw_min`` pulses. The nominal sent amount is subtracted
    from the BUFFER without reading the array back.

    Analog weight decay is deliberately unsupported. Decoupled
    decay acts on a weight value, whereas an ``AnalogContext`` BUFFER contains
    a residual rather than the current weight.  Analog parameters must thus be
    placed in groups with ``weight_decay=0``.  Digital parameter groups retain
    ordinary AdamW behavior, including weight decay.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.torch_step = AdamW.step
        self._validate_analog_groups()

    def regroup_param_groups(self, *args: Any) -> None:
        """Isolate analog contexts and validate their logical configuration."""

        super().regroup_param_groups(*args)
        self._validate_analog_groups()

    def _iter_analog_params(
        self,
    ) -> Iterator[Tuple[Dict[str, Any], AnalogContext]]:
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    yield group, param

    @staticmethod
    def _logical_scale(param: AnalogContext) -> float:
        scale = getattr(param, "logical_weight_scale", None)
        if scale is None:
            raise ValueError(
                "LogicalMpAdamW requires every AnalogContext to define "
                "logical_weight_scale before optimizer construction."
            )
        if isinstance(scale, Tensor):
            if scale.numel() != 1:
                raise ValueError("logical_weight_scale must be a scalar")
            scale = scale.detach().item()
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"logical_weight_scale must be finite and positive, got {scale}"
            )
        if not getattr(param, "logical_gradient_units", False):
            raise ValueError(
                "LogicalMpAdamW requires gradients accumulated in logical W units; "
                "the AnalogContext is missing logical_gradient_units=True."
            )
        return scale

    def _validate_analog_groups(self) -> None:
        for group, param in self._iter_analog_params():
            weight_decay = float(group.get("weight_decay", self.defaults["weight_decay"]))
            if weight_decay != 0.0:
                raise ValueError(
                    "LogicalMpAdamW does not support weight decay for "
                    "analog parameters. Put AnalogContext parameters in a group "
                    f"with weight_decay=0 (got {weight_decay})."
                )
            if param.use_torch_update:
                raise NotImplementedError(
                    "LogicalMpAdamW does not support AnalogContext.use_torch_update."
                )
            if param.use_indexed:
                raise NotImplementedError(
                    "LogicalMpAdamW does not support indexed analog updates."
                )
            self._logical_scale(param)

    @staticmethod
    def _has_complete_logical_state(state: Dict[str, Any]) -> bool:
        present = _REQUIRED_LOGICAL_STATE_KEYS.intersection(state)
        if present and present != _REQUIRED_LOGICAL_STATE_KEYS:
            missing = sorted(_REQUIRED_LOGICAL_STATE_KEYS.difference(state))
            raise ValueError(
                "Incomplete LogicalMpAdamW state; missing keys: " + ", ".join(missing)
            )
        return bool(present)

    def _validate_logical_state(
        self,
        param: AnalogContext,
        state: Dict[str, Any],
        *,
        require_state: bool,
    ) -> bool:
        complete = self._has_complete_logical_state(state)
        if not complete:
            if require_state and state:
                raise ValueError(
                    "An analog optimizer state contains Adam moments but no open-loop "
                    "logical residual metadata. It cannot be resumed safely with "
                    "LogicalMpAdamW."
                )
            return False

        version = int(state[_VERSION_KEY])
        if version != LOGICAL_MP_STATE_VERSION:
            raise ValueError(
                f"Unsupported LogicalMpAdamW state version {version}; "
                f"expected {LOGICAL_MP_STATE_VERSION}."
            )

        live_scale = self._logical_scale(param)
        saved_scale = float(state[_SCALE_KEY])
        if not math.isclose(saved_scale, live_scale, rel_tol=1e-7, abs_tol=0.0):
            raise ValueError(
                "Logical weight scale changed across resume: "
                f"checkpoint has {saved_scale}, model has {live_scale}."
            )

        residual = state[_RESIDUAL_KEY]
        if not isinstance(residual, Tensor):
            raise TypeError("logical_residual must be a tensor")
        if residual.shape != param.shape:
            raise ValueError(
                "Logical residual shape does not match its AnalogContext: "
                f"residual={tuple(residual.shape)}, context={tuple(param.shape)}."
            )
        if not torch.isfinite(residual).all():
            raise ValueError("Logical optimizer residual contains non-finite values")
        return True

    def _initialize_logical_state(
        self, param: AnalogContext, state: Dict[str, Any]
    ) -> None:
        # AdamW has initialized its standard state by this point. Because the
        # BUFFER started at zero, it now contains the first pending logical
        # increment. Keep a direct alias rather than a second model-sized copy.
        state[_VERSION_KEY] = LOGICAL_MP_STATE_VERSION
        state[_SCALE_KEY] = self._logical_scale(param)
        state[_RESIDUAL_KEY] = param.data
        state[_TRANSFER_STEPS_KEY] = 0

    def _prepare_states_for_torch_step(self) -> set[int]:
        """Validate existing states and identify never-stepped contexts."""

        fresh: set[int] = set()
        for _, param in self._iter_analog_params():
            state = self.state[param]
            if not state:
                fresh.add(id(param))
                continue
            self._validate_logical_state(param, state, require_state=True)
            residual = state[_RESIDUAL_KEY]
            if residual.data_ptr() != param.data.data_ptr():
                if not torch.equal(residual, param.data):
                    raise ValueError(
                        "AnalogContext BUFFER differs from the checkpointed logical "
                        "residual. Reload the optimizer state before continuing."
                    )
                state[_RESIDUAL_KEY] = param.data
        return fresh

    def _finish_states_after_torch_step(self, fresh: set[int]) -> None:
        for _, param in self._iter_analog_params():
            state = self.state[param]
            if not state:
                # No gradient: PyTorch AdamW did not initialize this parameter.
                continue
            if id(param) in fresh:
                self._initialize_logical_state(param, state)
            else:
                self._validate_logical_state(param, state, require_state=True)
                state[_RESIDUAL_KEY] = param.data

    @staticmethod
    def _update_tile(analog_tile: Any, physical_update: Tensor) -> None:
        """Apply an additive physical weight matrix using a full-rank update."""

        eye = getattr(analog_tile, "_logical_mp_eye_matrix", None)
        if (
            eye is None
            or eye.shape != (analog_tile.in_size, analog_tile.in_size)
            or eye.device != physical_update.device
            or eye.dtype != physical_update.dtype
        ):
            eye = torch.eye(
                analog_tile.in_size,
                device=physical_update.device,
                dtype=physical_update.dtype,
            )
            analog_tile._logical_mp_eye_matrix = eye
        # AIHWKit uses W <- W - d^T x, hence the minus sign makes this an
        # additive update by ``physical_update``.
        analog_tile.update(eye, -physical_update.transpose(0, 1))

    def _transfer_one(self, param: AnalogContext, state: Dict[str, Any]) -> None:
        """Open-loop transfer and nominal BUFFER settlement."""

        analog_tile = param.analog_tile
        scale = self._logical_scale(param)
        residual = param.data

        analog_tile.set_learning_rate(1.0)
        rpu_device = self._get_RPU_device(analog_tile.rpu_config)
        if rpu_device is None:
            raise TypeError(
                "LogicalMpAdamW cannot determine the physical update device from "
                f"{type(analog_tile.rpu_config.device).__name__}."
            )

        if isinstance(rpu_device, FloatingPointDevice):
            logical_transfer = residual.clone()
            physical_update = logical_transfer.div(scale)
        else:
            dw_min = float(rpu_device.dw_min)
            if not math.isfinite(dw_min) or dw_min < 0.0:
                raise ValueError(
                    f"Device dw_min must be finite and non-negative, got {dw_min}"
                )

            if dw_min == 0.0:
                # AIHWKit's infinite-granularity pulsed-device mode.
                logical_transfer = residual.clone()
                physical_update = logical_transfer.div(scale)
            else:
                max_pulses = int(analog_tile.rpu_config.update.desired_bl)
                if max_pulses <= 0:
                    raise ValueError(
                        f"update.desired_bl must be positive, got {max_pulses}"
                    )
                logical_granularity = scale * dw_min
                desired_pulses = residual.div(logical_granularity).trunc()
                available_pulses = desired_pulses.clamp(
                    min=-max_pulses, max=max_pulses
                )
                physical_update = available_pulses.mul(dw_min)
                logical_transfer = available_pulses.mul(logical_granularity)

        self._update_tile(analog_tile, physical_update)
        analog_tile.post_update_step()
        # Open-loop semantics: settle the nominal command, not a readback of
        # the state-dependent/noisy/clipped physical response.
        residual.sub_(logical_transfer)
        state[_RESIDUAL_KEY] = residual
        state[_TRANSFER_STEPS_KEY] = int(state[_TRANSFER_STEPS_KEY]) + 1

    @no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None, **kwargs: Any
    ) -> Optional[float]:
        """Run standard AdamW in W units, then transfer open-loop commands."""

        self._validate_analog_groups()
        self.prepare_grad()
        fresh = self._prepare_states_for_torch_step()

        assert self.torch_step is not None, "LogicalMpAdamW requires AdamW.step"
        result = self.torch_step(self, closure, **kwargs)
        self._finish_states_after_torch_step(fresh)

        if self.freeze_analog_weight:
            return result

        for _, param in self._iter_analog_params():
            state = self.state[param]
            if state:
                self._transfer_one(param, state)
        return result

    def state_dict(self) -> Dict[str, Any]:
        """Return Adam state plus the pending logical open-loop BUFFER."""

        for _, param in self._iter_analog_params():
            state = self.state.get(param)
            if not state:
                continue
            self._validate_logical_state(param, state, require_state=True)
            state[_RESIDUAL_KEY] = param.data
        return super().state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore moments and pending logical increments without weight reads."""

        super().load_state_dict(state_dict)
        self._validate_analog_groups()

        for _, param in self._iter_analog_params():
            state = self.state.get(param)
            if not state:
                param.data.zero_()
                continue
            self._validate_logical_state(param, state, require_state=True)
            residual = state[_RESIDUAL_KEY]
            param.data.copy_(residual.to(device=param.device, dtype=param.dtype))
            state[_RESIDUAL_KEY] = param.data

    def logical_state(self, param: AnalogContext) -> Dict[str, Any]:
        """Return logical optimizer metadata without reading array weights."""

        if not isinstance(param, AnalogContext):
            raise TypeError("logical_state expects an AnalogContext parameter")
        state = self.state.get(param)
        if not state or not self._validate_logical_state(
            param, state, require_state=True
        ):
            raise ValueError("AnalogContext has not received an optimizer step")
        return {
            "version": int(state[_VERSION_KEY]),
            "scale": float(state[_SCALE_KEY]),
            "residual": state[_RESIDUAL_KEY],
            "transfer_steps": int(state[_TRANSFER_STEPS_KEY]),
        }


__all__ = ["LOGICAL_MP_STATE_VERSION", "LogicalMpAdamW"]
