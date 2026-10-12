"""Shared mapped-weight coordinates, residuals, and open-loop AIMC writes.

Concrete optimizers supply the digital update rule. This base owns scale
validation, residual storage, quantized tile transfers, and checkpoint state.
It composes with a PyTorch optimizer through cooperative inheritance.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Adafactor, LBFGS, Optimizer
from torch.autograd import no_grad
from torch.nn.parallel import DistributedDataParallel

from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs.devices import FloatingPointDevice, PulsedDevice
from aihwkit.simulator.configs.compounds import UnitCell
from aihwkit.simulator.parameters.enums import AnalogContextDataViewMode, PulseType

# Version 3 keeps s exclusively in model tile.mapping_scales.
# Historical formats must be repaired offline before loading.
MP_STATE_VERSION = 3

_VERSION_KEY = "mp_version"
_RESIDUAL_KEY = "mp_residual"
_TRANSFER_STEPS_KEY = "mp_transfer_steps"

_REQUIRED_MP_STATE_KEYS = {
    _VERSION_KEY,
    _RESIDUAL_KEY,
    _TRANSFER_STEPS_KEY,
}


class MpOptimizerMixin(Optimizer):
    """Plug-in adapter for PyTorch optimization and analog programming.

    Compose as ``class MyMpOptimizer(MpOptimizerMixin, MyTorchOptimizer)``
    and set ``digital_optimizer_cls = MyTorchOptimizer``, or use
    ``as_logical_mp(MyTorchOptimizer)``. No update recurrence lives here.

    The digital rule updates the residual directly through a proxy that shares
    its storage. Its update increment must depend only on gradients and rule
    state, not the current weight value. Weight decay is disabled. No digital
    weight copy or initialization read from the analog array is needed.
    Analog parameter groups are isolated automatically during construction.
    Weight decay must remain zero throughout training; configuration checks
    run at construction, group registration and checkpoint boundaries.

    The tile stores W_hat; mapped-weight layers represent W = s * W_hat. A concrete
    rule accumulates its signed mapped-weight update in a digital residual, then
    calls _transfer_pending_updates() to send physical commands. Settlement
    subtracts nominal sent increments and never uses weight readback.
    """

    allow_indexed_mp_updates = True
    digital_optimizer_cls = None

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs: Any) -> Optional[float]:
        """Delegate one digital step and send its pending mapped-weight increments."""
        # Evaluate once against the live analog model, before collecting its
        # gradients. The digital rule never evaluates a stale proxy-weight model.
        self._ddp_require_synchronized_gradients()
        loss = None
        if closure is not None:
            if self._ddp_prepared:
                raise NotImplementedError(
                    "Evaluate and synchronize the closure before a DDP MpOptimizerMixin step"
                )
            with torch.enable_grad():
                loss = closure()
        self._prepare_mp_gradients()
        original_groups = [group["params"] for group in self.param_groups]
        adapted = []
        try:
            for group in self.param_groups:
                digital_params = []
                for param in group["params"]:
                    if not isinstance(param, AnalogContext):
                        digital_params.append(param)
                        continue
                    if param.grad is None:
                        # Preserve this tile's BUFFER until it has a gradient.
                        continue
                    state = self.state[param]
                    if _VERSION_KEY not in state:
                        self._initialize_mp_state(param, state)
                    target = param.data
                    proxy = self._mp_proxies.get(param)
                    if proxy is None:
                        proxy = self._mp_proxies[param] = nn.Parameter(target.detach())
                    proxy.data = target
                    proxy.grad = param.grad
                    metadata = {key: value for key, value in state.items() if key in _REQUIRED_MP_STATE_KEYS}
                    rule_state = {
                        key: value for key, value in state.items() if key not in _REQUIRED_MP_STATE_KEYS
                    }
                    self.state[proxy] = rule_state
                    adapted.append((param, proxy, metadata))
                    digital_params.append(proxy)
                group["params"] = digital_params

            digital_step = self.digital_optimizer_cls.step
            # Optimizer profiling hooks belong to the public adapter step.
            # Avoid invoking them a second time if the native class was patched.
            if getattr(digital_step, "hooked", False):
                digital_step = digital_step.__wrapped__
            result = digital_step(self, **kwargs)
        finally:
            for group, params in zip(self.param_groups, original_groups):
                group["params"] = params
            for param, proxy, metadata in adapted:
                state = self.state.pop(proxy)
                state.update(metadata)
                buffer = param.data
                # Native rules normally update the shared BUFFER in place.
                # Only copy back if a rule replaced its proxy's storage.
                if proxy.data_ptr() != buffer.data_ptr():
                    buffer.copy_(proxy.detach())
                state[_RESIDUAL_KEY] = buffer
                self.state[param] = state
                proxy.grad = None

        self._transfer_pending_updates([param for param, _, _ in adapted])
        return loss if closure is not None else result

    def _ddp_step_protocol_is_supported(self) -> bool:
        step_impl = type(self).step
        reference = MpOptimizerMixin.step
        while hasattr(step_impl, "__wrapped__"):
            step_impl = step_impl.__wrapped__
        while hasattr(reference, "__wrapped__"):
            reference = reference.__wrapped__
        return step_impl is reference

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self.digital_optimizer_cls is None:
            raise TypeError("Compose MpOptimizerMixin with a PyTorch optimizer or use as_logical_mp()")
        if issubclass(self.digital_optimizer_cls, Adafactor):
            raise NotImplementedError(
                "PyTorch Adafactor depends on current weight RMS; MpOptimizerMixin exposes residuals, not weights"
            )
        if issubclass(self.digital_optimizer_cls, LBFGS):
            raise NotImplementedError(
                "MpOptimizerMixin does not support optimizers that reevaluate closures, such as LBFGS"
            )
        if kwargs.get("differentiable", False):
            raise NotImplementedError(
                "Differentiable optimizer steps are not supported by analog programming"
            )
        self._mp_proxies = {}
        super().__init__(*args, **kwargs)
        self._validate_weight_decay()
        self.freeze_analog_weight = False
        self._ddp_prepared = False
        self._ddp_rank = 0
        self._ddp_world_size = 1
        self._ddp_gradients_synchronized = False
        self._ddp_analog_params = []
        self._ddp_analog_param_names = []
        self.check_update_config()
        for _, param in self._iter_analog_params():
            self._initialize_analog_context(param, reset_buffer=True)
        self.regroup_param_groups()
        # Some native optimizers (Adagrad) allocate rule state in __init__.
        for _, param in self._iter_analog_params():
            state = self.state.get(param)
            if state and not self._has_complete_mp_state(state):
                self._initialize_mp_state(param, state)

    def regroup_param_groups(self, *args: Any) -> None:
        """Isolate analog contexts and validate their mapped-weight configuration."""

        if args and hasattr(args[0], "named_parameters"):
            try:
                self._param_name_by_id = {
                    id(param): name for name, param in args[0].named_parameters()
                }
            except Exception:
                self._param_name_by_id = {}

        analog_groups = []
        for group in self.param_groups:
            digital_params = []
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.set_learning_rate(group["lr"])
                    analog_groups.append(
                        {"params": [param], **{k: v for k, v in group.items() if k != "params"}}
                    )
                else:
                    digital_params.append(param)
            group["params"] = digital_params
        self.param_groups = [group for group in self.param_groups if group["params"]]
        for group in analog_groups:
            self.add_param_group(group)
        self._validate_analog_groups()

    def set_learning_rate(self, learning_rate: float = 0.1) -> None:
        """Set the learning rate on parameter groups and their analog tiles."""
        for group in self.param_groups:
            group["lr"] = learning_rate
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.set_learning_rate(learning_rate)

    def _iter_analog_params(
        self,
    ) -> Iterator[Tuple[Dict[str, Any], AnalogContext]]:
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    yield group, param

    def _validate_analog_groups(self) -> None:
        self._validate_weight_decay()
        for group, param in self._iter_analog_params():
            if param.use_torch_update:
                raise NotImplementedError(
                    "MpOptimizerMixin does not support AnalogContext.use_torch_update."
                )
            if not getattr(param, "mapped_gradient_units", False):
                raise ValueError("MpOptimizerMixin requires mapped_gradient_units=True")
            analog_tile = param.analog_tile
            scale = analog_tile.get_mapping_scales()
            if (
                scale is None
                or scale.numel() not in (1, analog_tile.out_size)
                or not torch.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError("mapping_scales must be finite, positive and match tile rows")
            alpha = analog_tile.out_scaling_alpha
            if alpha is not None and (
                alpha.numel() not in (1, analog_tile.out_size)
                or not torch.isfinite(alpha).all()
            ):
                raise ValueError("out_scaling_alpha must be finite and match tile rows")
            lr_scale = float(analog_tile.mapping_lr_scale)
            if not math.isfinite(lr_scale) or lr_scale <= 0.0:
                raise ValueError("mapping_lr_scale must be finite and positive")
            rpu_device = self._get_RPU_device(analog_tile.rpu_config)
            if rpu_device is None:
                raise TypeError("MpOptimizerMixin cannot determine the physical update device")
            if not isinstance(rpu_device, FloatingPointDevice):
                dw_min = float(rpu_device.dw_min)
                if not math.isfinite(dw_min) or dw_min < 0.0:
                    raise ValueError("Device dw_min must be finite and non-negative")
                if dw_min > 0.0 and int(analog_tile.rpu_config.update.desired_bl) <= 0:
                    raise ValueError("update.desired_bl must be positive")

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        self._validate_weight_decay([param_group])
        super().add_param_group(param_group)
        if hasattr(self, "_ddp_prepared"):
            for param in self.param_groups[-1]["params"]:
                if isinstance(param, AnalogContext):
                    self._initialize_analog_context(param)
            self._validate_analog_groups()

    @staticmethod
    def _has_complete_mp_state(state: Dict[str, Any]) -> bool:
        present = _REQUIRED_MP_STATE_KEYS.intersection(state)
        if present and present != _REQUIRED_MP_STATE_KEYS:
            missing = sorted(_REQUIRED_MP_STATE_KEYS.difference(state))
            raise ValueError("Incomplete MpOptimizerMixin state; missing keys: " + ", ".join(missing))
        return bool(present)

    def _validate_mp_state(
        self,
        param: AnalogContext,
        state: Dict[str, Any],
        *,
        require_state: bool,
    ) -> bool:
        complete = self._has_complete_mp_state(state)
        if not complete:
            if require_state and state:
                raise ValueError(
                    "An analog optimizer state contains rule state but no open-loop "
                    "mapped-weight residual metadata. It cannot be resumed safely with "
                    "MpOptimizerMixin."
                )
            return False

        version = int(state[_VERSION_KEY])
        if version != MP_STATE_VERSION:
            raise ValueError(
                f"Unsupported MpOptimizerMixin state version {version}; "
                f"expected {MP_STATE_VERSION}."
            )

        residual = state[_RESIDUAL_KEY]
        if not isinstance(residual, Tensor):
            raise TypeError("mp_residual must be a tensor")
        if residual.shape != param.shape:
            raise ValueError(
                "MP residual shape does not match its AnalogContext: "
                f"residual={tuple(residual.shape)}, context={tuple(param.shape)}."
            )
        if not torch.isfinite(residual).all():
            raise ValueError("MP optimizer residual contains non-finite values")
        return True

    def _initialize_mp_state(self, param: AnalogContext, state: Dict[str, Any]) -> None:
        # All analog parameters use the context's writable BUFFER as H.
        state[_VERSION_KEY] = MP_STATE_VERSION
        state[_RESIDUAL_KEY] = param.data
        state[_TRANSFER_STEPS_KEY] = 0
        self._validate_mp_state(param, state, require_state=True)

    @staticmethod
    @torch.no_grad()
    def _update_mapping_tile(analog_tile: Any, mapped_increment: Tensor) -> None:
        """Submit delta_W; native pre_update converts it to physical tile units."""
        analog_tile.set_learning_rate(1.0)
        eye = getattr(analog_tile, "_mp_eye_matrix", None)
        size = analog_tile.in_size
        if (
            eye is None
            or eye.shape != (size, size)
            or eye.device != mapped_increment.device
            or eye.dtype != mapped_increment.dtype
        ):
            eye = torch.eye(size, device=mapped_increment.device, dtype=mapped_increment.dtype)
            analog_tile._mp_eye_matrix = eye
        analog_tile.update(eye, -mapped_increment.T)

    def _transfer_one(self, param: AnalogContext, state: Dict[str, Any]) -> None:
        """Open-loop transfer and nominal BUFFER settlement."""

        analog_tile = param.analog_tile
        residual = param.data

        write_tile = self._update_mapping_tile
        rpu_device = self._get_RPU_device(analog_tile.rpu_config)
        if isinstance(rpu_device, FloatingPointDevice) or rpu_device.dw_min == 0.0:
            transfer_increment = residual.clone()
            update_command = transfer_increment
        else:
            dw_min = float(rpu_device.dw_min)
            max_pulses = int(analog_tile.rpu_config.update.desired_bl)
            scale = analog_tile.get_mapping_scales().detach().reshape(-1, 1)
            write_granularity = scale.abs() * dw_min
            desired_pulses = residual.div(write_granularity).trunc()
            available_pulses = desired_pulses.clamp(min=-max_pulses, max=max_pulses)
            transfer_increment = available_pulses.mul(write_granularity)
            # Native pre_update owns the only W -> W_hat conversion.
            update_command = transfer_increment

        if self._ddp_is_update_rank():
            write_tile(analog_tile, update_command)
        # Open-loop semantics: settle the nominal command, not a readback of
        # the state-dependent/noisy/clipped physical response.
        residual.sub_(transfer_increment)
        state[_RESIDUAL_KEY] = residual
        state[_TRANSFER_STEPS_KEY] = int(state[_TRANSFER_STEPS_KEY]) + 1

    def state_dict(self) -> Dict[str, Any]:
        """Return rule-specific state plus the pending mapped-weight open-loop BUFFER."""

        self._validate_weight_decay()
        for _, param in self._iter_analog_params():
            state = self.state.get(param)
            if not state:
                continue
            state[_RESIDUAL_KEY] = param.data
        return super().state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore moments and pending mapped-weight increments without weight reads."""

        self._validate_weight_decay(state_dict["param_groups"])
        for group, saved_group in zip(self.param_groups, state_dict["param_groups"]):
            for param, key in zip(group["params"], saved_group["params"]):
                if isinstance(param, AnalogContext):
                    self._validate_mp_state(
                        param, state_dict["state"].get(key, {}), require_state=True
                    )
        super().load_state_dict(state_dict)
        self._validate_analog_groups()

        for _, param in self._iter_analog_params():
            state = self.state.get(param)
            if not state:
                param.data.zero_()
                continue
            self._validate_mp_state(param, state, require_state=True)
            residual = state[_RESIDUAL_KEY]
            param.data.copy_(residual.to(device=param.device, dtype=param.dtype))
            state[_RESIDUAL_KEY] = param.data

    def mp_state(self, param: AnalogContext) -> Dict[str, Any]:
        """Return BUFFER residual and transfer count; scales live on the tile."""

        if not isinstance(param, AnalogContext):
            raise TypeError("mp_state expects an AnalogContext parameter")
        state = self.state.get(param)
        if not state or _VERSION_KEY not in state:
            raise ValueError("AnalogContext has not received an optimizer step")
        return {
            "residual": state[_RESIDUAL_KEY],
            "transfer_steps": int(state[_TRANSFER_STEPS_KEY]),
        }

    def _prepare_mp_gradients(self) -> None:
        """Obtain synchronized gradients already in W = s * W_hat units."""
        self._ddp_require_synchronized_gradients()
        self.prepare_grad()

    def _transfer_pending_updates(self, active_params: list[AnalogContext]) -> None:
        """Write tiles updated this step, then advance every tile once."""
        if not self.freeze_analog_weight:
            for param in active_params:
                self._transfer_one(param, self.state[param])
            if self._ddp_is_update_rank():
                for _, param in self._iter_analog_params():
                    param.analog_tile.post_update_step()
        self._ddp_finish_step()

    @staticmethod
    def _initialize_analog_context(param: AnalogContext, *, reset_buffer: bool = False) -> None:
        """Use BUFFER storage independently of the gradient collection path."""
        if param.analog_tile.input_range is not None:
            raise NotImplementedError("MP optimization with input_range is not supported")
        param.mapped_gradient_units = True
        if param.analog_tile.get_mapping_scales() is None:
            param.analog_tile.set_mapping_scales(
                torch.ones(1, device=param.device, dtype=param.dtype)
            )
        param.set_digital_grad_accumulation(True, use_pre_update=False)
        if reset_buffer or param.data_view_mode != AnalogContextDataViewMode.BUFFER:
            param.enable_buffer()

    def _validate_weight_decay(self, groups=None) -> None:
        """Reject coupled or decoupled decay in every wrapped parameter group."""
        for index, group in enumerate(self.param_groups if groups is None else groups):
            decay = group.get("weight_decay", self.defaults.get("weight_decay", 0.0))
            if float(decay) != 0.0:
                raise NotImplementedError(
                    f"{type(self).__name__}: weight decay is temporarily disabled "
                    f"(parameter group {index} has weight_decay={decay}). "
                    "Set weight_decay=0.0 explicitly in all parameter groups."
                )

    def zero_grad(self, *args: Any, **kwargs: Any):
        super().zero_grad(*args, **kwargs)
        self._ddp_gradients_synchronized = False
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.reset()  # remove the x and d signals

    @staticmethod
    def _ddp_to_cpu(value: Any) -> Any:
        """Clone a nested optimizer/tile payload onto CPU."""

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: MpOptimizerMixin._ddp_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [MpOptimizerMixin._ddp_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(MpOptimizerMixin._ddp_to_cpu(item) for item in value)
        return value

    @staticmethod
    def _ddp_collective_device() -> torch.device:
        if torch.distributed.get_backend() == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL DDP requires an available CUDA device")
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _ddp_broadcast_object(self, value: Any) -> Any:
        objects = [value if self._ddp_rank == 0 else None]
        torch.distributed.broadcast_object_list(
            objects, src=0, device=self._ddp_collective_device()
        )
        return objects[0]

    @staticmethod
    def _ddp_capture_tile_state(param: AnalogContext) -> Dict[str, Any]:
        """Capture every AIHWKit component used by checkpoint restore."""

        analog_tile = param.analog_tile
        simulator_tile = getattr(analog_tile, "tile", None)
        required = (
            "get_weights",
            "set_weights",
            "get_hidden_parameters",
            "set_hidden_parameters",
            "get_hidden_parameter_names",
            "get_learning_rate",
            "set_learning_rate",
            "dump_extra",
            "load_extra",
        )
        missing = [name for name in required if not callable(getattr(simulator_tile, name, None))]
        if missing:
            raise RuntimeError(
                "Cannot guarantee DDP synchronization for AIHWKit tile "
                f"{type(analog_tile).__name__}: missing simulator state APIs " + ", ".join(missing)
            )

        extra = simulator_tile.dump_extra()
        if not isinstance(extra, dict):
            raise RuntimeError(
                "Cannot guarantee DDP synchronization: tile.dump_extra() did not "
                f"return a dictionary for {type(analog_tile).__name__}."
            )
        try:
            context_buffer = param.data.detach().cpu().clone()
        except RuntimeError as exc:
            raise RuntimeError(
                "DDP mixed-precision synchronization requires AnalogContext BUFFER "
                "mode so the mapped-weight residual can be broadcast."
            ) from exc
        return {
            "weights": simulator_tile.get_weights().detach().cpu().clone(),
            "hidden_parameters": (simulator_tile.get_hidden_parameters().detach().cpu().clone()),
            "hidden_parameter_names": list(simulator_tile.get_hidden_parameter_names()),
            "extra": MpOptimizerMixin._ddp_to_cpu(extra),
            "learning_rate": float(simulator_tile.get_learning_rate()),
            "context_buffer": context_buffer,
        }

    @staticmethod
    def _ddp_restore_tile_state(param: AnalogContext, state: Dict[str, Any]) -> None:
        analog_tile = param.analog_tile
        simulator_tile = analog_tile.tile
        live_names = list(simulator_tile.get_hidden_parameter_names())
        saved_names = list(state["hidden_parameter_names"])
        if live_names != saved_names:
            raise RuntimeError(
                "Cannot restore rank-0 AIHWKit tile state: hidden parameter names "
                f"differ (live={live_names}, rank0={saved_names})."
            )

        # Match AIHWKit's own __setstate__ order. Hidden parameters include
        # device-to-device variation and other state invisible in the weight
        # tensor; dump_extra() carries counters and compound-device state.
        simulator_tile.set_hidden_parameters(state["hidden_parameters"])
        simulator_tile.set_weights(state["weights"])
        simulator_tile.set_learning_rate(float(state["learning_rate"]))
        simulator_tile.load_extra(state["extra"], True)
        param.data.copy_(state["context_buffer"].to(device=param.device, dtype=param.dtype))

    def _ddp_broadcast_optimizer_state(self) -> None:
        state = None
        if self._ddp_rank == 0:
            state = self._ddp_to_cpu(self.state_dict())
        state = self._ddp_broadcast_object(state)
        if self._ddp_rank != 0:
            self.load_state_dict(state)

    def _ddp_broadcast_tile_states(self) -> None:
        for param in self._ddp_analog_params:
            state = None
            if self._ddp_rank == 0:
                state = self._ddp_capture_tile_state(param)
            state = self._ddp_broadcast_object(state)
            if self._ddp_rank != 0:
                self._ddp_restore_tile_state(param, state)

    def prepare_for_ddp(self, model: Any) -> None:
        """Broadcast rank-0 state before constructing PyTorch DDP.

        Construction automatically regroups parameters; checkpoint state must
        already be loaded. AnalogContexts are then excluded from the standard
        reducer so their gradients cannot be reduced twice.
        """

        if self._ddp_prepared:
            raise RuntimeError("prepare_for_ddp() may only be called once")
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "prepare_for_ddp() requires an initialized torch.distributed " "process group"
            )
        if not self._ddp_step_protocol_is_supported():
            raise RuntimeError(
                f"{type(self).__name__} overrides optimizer.step() without "
                "integrating the centralized AIHWKit DDP tile-update protocol."
            )

        analog_params = [
            param
            for group in self.param_groups
            for param in group["params"]
            if isinstance(param, AnalogContext)
        ]
        if not analog_params:
            raise RuntimeError("AIHWKit DDP preparation found no AnalogContext parameters")
        if len({id(param) for param in analog_params}) != len(analog_params):
            raise RuntimeError("An AnalogContext occurs in more than one optimizer group")

        name_by_id = {id(param): name for name, param in model.named_parameters()}
        if any(id(param) not in name_by_id for param in analog_params):
            raise RuntimeError(
                "Every optimizer AnalogContext must be registered in the model " "before DDP"
            )
        for param in analog_params:
            if param.use_torch_update:
                raise RuntimeError(
                    "The centralized training-tile DDP protocol does not support "
                    "AnalogContext.use_torch_update."
                )
            if not getattr(param, "digital_grad_accumulation", False):
                raise RuntimeError(
                    "The centralized training-tile DDP protocol requires "
                    "backward-time digital AnalogContext gradient accumulation. "
                    "Indexed/deferred X/D reconstruction is not supported."
                )

        self._ddp_rank = torch.distributed.get_rank()
        self._ddp_world_size = torch.distributed.get_world_size()
        if self._ddp_world_size < 2:
            raise RuntimeError("AIHWKit DDP protocol requires world_size >= 2")
        self._ddp_analog_params = analog_params
        self._ddp_analog_param_names = [name_by_id[id(param)] for param in analog_params]

        self._ddp_broadcast_optimizer_state()
        self._ddp_broadcast_tile_states()

        ignored = set(getattr(model, "_ddp_params_and_buffers_to_ignore", set()))
        ignored.update(self._ddp_analog_param_names)
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, ignored)
        self._ddp_prepared = True
        self._ddp_gradients_synchronized = False

    def ddp_synchronize_after_load(self) -> None:
        """Broadcast rank-0 optimizer and tile state after an in-place resume."""

        if not self._ddp_prepared:
            raise RuntimeError("ddp_synchronize_after_load() requires prior prepare_for_ddp()")
        self._ddp_broadcast_optimizer_state()
        self._ddp_broadcast_tile_states()
        self._ddp_gradients_synchronized = False

    @no_grad()
    def ddp_synchronize_gradients(self, ddp_model: Any) -> bool:
        """Average AnalogContext gradients exactly once for this step."""

        if not self._ddp_prepared:
            raise RuntimeError("ddp_synchronize_gradients() called before prepare_for_ddp()")
        sync_required = getattr(ddp_model, "require_backward_grad_sync", None)
        if sync_required is None:
            raise TypeError(
                "ddp_synchronize_gradients() expects a " "DistributedDataParallel model"
            )
        if not bool(sync_required):
            return False
        if self._ddp_gradients_synchronized:
            raise RuntimeError(
                "AnalogContext gradients were already synchronized for this " "optimizer step"
            )

        collective_device = self._ddp_collective_device()
        for param in self._ddp_analog_params:
            present = torch.tensor(
                [1 if param.grad is not None else 0],
                device=collective_device,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(present, op=torch.distributed.ReduceOp.SUM)
            if int(present.item()) == 0:
                continue
            grad = param.grad
            if grad is None:
                grad = torch.zeros(tuple(param.shape), device=param.device, dtype=param.dtype)
            elif tuple(grad.shape) != tuple(param.shape):
                raise RuntimeError(
                    "AnalogContext gradient shape differs from its mapped-weight shape: "
                    f"grad={tuple(grad.shape)}, context={tuple(param.shape)}"
                )
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
            grad.div_(self._ddp_world_size)
            param.grad = grad

        self._ddp_gradients_synchronized = True
        return True

    def _ddp_require_synchronized_gradients(self) -> None:
        if self._ddp_prepared and not self._ddp_gradients_synchronized:
            raise RuntimeError(
                "optimizer.step() would update an AIHWKit training tile before "
                "AnalogContext gradients were synchronized. Call "
                "optimizer.ddp_synchronize_gradients(ddp_model) after the final "
                "accumulation micro-step."
            )

    def _ddp_is_update_rank(self) -> bool:
        return not self._ddp_prepared or self._ddp_rank == 0

    def _ddp_finish_step(self) -> None:
        if not self._ddp_prepared:
            return
        self._ddp_broadcast_tile_states()
        self._ddp_gradients_synchronized = False

    def check_update_config(self):
        """Configure deterministic writes; Python owns pulse quantization."""
        for _, param in self._iter_analog_params():
            tile = param.analog_tile
            config = tile.rpu_config
            if isinstance(config.device, FloatingPointDevice):
                continue
            changed = False
            for name, required in (
                ("pulse_type", PulseType.DETERMINISTIC_IMPLICIT),
                ("update_management", True),
            ):
                if getattr(config.update, name) != required:
                    warnings.warn(f"Setting MP update.{name} to {required}", UserWarning)
                    setattr(config.update, name, required)
                    changed = True
            if changed:
                tile.to(config)

    def get_processed_x_d_input(self, analog_ctx):
        analog_tile = analog_ctx.analog_tile
        runtime = analog_tile.get_runtime()
        if analog_ctx.use_indexed:
            # Native accumulation handles the exact indexed geometry. This
            # fallback is only for explicitly retained legacy X/D traces.
            previous_grad = analog_ctx.grad
            analog_ctx.grad = None
            try:
                for x, d in zip(analog_ctx.analog_input, analog_ctx.analog_grad_output):
                    analog_ctx.accumulate_digital_gradient(
                        x.to(analog_tile.device), d.to(analog_tile.device)
                    )
                return [analog_ctx.grad]
            finally:
                analog_ctx.grad = previous_grad

        # A shared Conv can be called at several pyramid resolutions in one
        # autograd graph (FPN/RPN).  Its unfolded traces then differ in their
        # sequence dimension and cannot be concatenated directly.  Process
        # each call independently and return additive weight gradients; this
        # is exactly the same sum as concatenation when shapes do agree.
        gradients = []
        for x_input, d_input in zip(analog_ctx.analog_input, analog_ctx.analog_grad_output):
            if runtime.offload_input:
                x_input = x_input.to(analog_tile.device)
            if runtime.offload_gradient:
                d_input = d_input.to(analog_tile.device)
            gradients.append(self.compute_gradient(x_input, d_input))
        return gradients

    def _get_RPU_device(self, rpu_config):
        if isinstance(rpu_config.device, PulsedDevice):
            return rpu_config.device
        elif isinstance(rpu_config.device, UnitCell):
            return rpu_config.device.unit_cell_devices[0]
        elif isinstance(rpu_config.device, FloatingPointDevice):
            return rpu_config.device
        else:
            return None  # floating point

    @no_grad()
    def compute_gradient(self, x_input, d_input):
        """Compute gradient via digital MAC operations.

        Args:
            x_input: input activations
            d_input: gradient w.r.t. output activations

        Returns:
            The computed gradient.
        """
        # Indexed Conv2d returns a weight gradient per saved forward batch.
        if isinstance(x_input, list):
            if not x_input:
                raise RuntimeError("Indexed AnalogConv has no saved X/D batches.")
            grad = None
            for batch_grad in x_input:
                grad = batch_grad if grad is None else grad.add_(batch_grad)
            return grad

        # Compute gradient as average of outer products over batch
        # W: (N, M)
        # x_input: (B, M), d_input: (B, N)
        # gradient: (M, N) = sum(d_i ⊗ x_i) for i in batch
        # The avearage over the batchsize is implemented in the training code
        #   here we only sum up the gradient
        # Feature maps from maintained vision models can be non-contiguous
        # (for example after FPN interpolation).  ``reshape`` preserves the
        # mathematical flattening while materialising a contiguous copy only
        # when necessary; ``view`` would incorrectly reject those valid
        # traces.
        grad = torch.matmul(
            d_input.reshape(-1, d_input.size(-1)).transpose(0, 1),
            x_input.reshape(-1, x_input.size(-1)),
        )

        return grad

    @no_grad()
    def prepare_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param

                if analog_ctx.digital_grad_use_pre_update:
                    raise ValueError(
                        "MP optimizers require use_pre_update=False; gradients must target W = s * W_hat"
                    )

                if (
                    getattr(analog_ctx, "digital_grad_accumulation", False)
                    and param.grad is not None
                ):
                    # Linear and indexed Conv both accumulate natively in W units.
                    continue

                # Legacy path: build the gradient from retained x and d traces.
                if not analog_ctx.has_gradient():
                    # A converted composite module can retain a wrapper context
                    # without owning a trainable tile. It has no x/d trace and
                    # must simply be skipped, like an unused torch parameter.
                    continue

                processed = self.get_processed_x_d_input(analog_ctx)
                if isinstance(processed, list):
                    grad = self.compute_gradient(processed, None)
                else:
                    x_input, d_input = processed
                    grad = self.compute_gradient(x_input, d_input)

                param.grad = grad


@lru_cache(maxsize=None)
def as_logical_mp(optimizer_cls: type, *, name: Optional[str] = None) -> type:
    """Create an MP optimizer class from an ordinary PyTorch optimizer class.

    Example: ``MpAdam = as_logical_mp(torch.optim.Adam)`` followed by
    ``MpAdam(model.parameters(), lr=1e-3, weight_decay=0.0)``. The update
    increment must be independent of the current weight value; analog proxies
    expose pending residuals, not weights. Nonzero weight decay is rejected.
    The rule must operate on param_groups and perform one step with supplied
    gradients. Multi-evaluation closures (LBFGS), differentiable steps, and
    changing the parameter-group topology during step are unsupported.
    """
    if not isinstance(optimizer_cls, type) or not issubclass(optimizer_cls, Optimizer):
        raise TypeError("optimizer_cls must be a torch.optim.Optimizer class")
    if issubclass(optimizer_cls, MpOptimizerMixin):
        return optimizer_cls
    class_name = name or "Mp" + optimizer_cls.__name__
    existing = globals().get(class_name)
    if existing is not None:
        if getattr(existing, "digital_optimizer_cls", None) is optimizer_cls:
            return existing
        raise ValueError("Optimizer name is already registered; supply a unique name")
    result = type(
        class_name,
        (MpOptimizerMixin, optimizer_cls),
        dict(
            __module__=__name__,
            digital_optimizer_cls=optimizer_cls,
        ),
    )
    globals()[class_name] = result
    return result


__all__ = ["MP_STATE_VERSION", "MpOptimizerMixin", "as_logical_mp"]
