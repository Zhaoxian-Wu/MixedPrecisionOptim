import os
import random
import time
import warnings
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Optional

import torch
from torch import cat
from torch.autograd import no_grad
from torch.nn.functional import cosine_similarity

from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs.compounds import UnitCell
from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice, PulsedDevice
from aihwkit.simulator.parameters.enums import PulseType

class AnalogOptimizerMixin:
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """

    def regroup_param_groups(self, *_: Any) -> None:
        """Reorganize the parameter groups, isolating analog layers.

        Update the `param_groups` of the optimizer, moving the parameters for
        each analog layer to a new single group.
        """
        # Cache param id -> name mapping when a model/module is passed.
        # `regroup_param_groups(model)` is called from training scripts.
        if len(_) > 0 and hasattr(_[0], "named_parameters"):
            try:
                self._param_name_by_id = {
                    id(param): name for name, param in _[0].named_parameters()
                }
            except Exception:
                self._param_name_by_id = {}

        # Create the new param groups.
        analog_param_groups = []
        rm_group_lst = []
        for group in self.param_groups:  # type: ignore[has-type]
            rm_lst = []
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.set_learning_rate(
                        # self.defaults["lr"]  # type: ignore[attr-defined]
                        # TODO:
                        group["lr"]  # type: ignore[attr-defined]
                    )
                    analog_param_groups.append({"params": [param], **{
                        k: v for k, v in group.items() if k != "params"
                    }})
                    rm_lst.append(id(param))

            group["params"] = [p for p in group["params"] if id(p) not in rm_lst]

            if len(group["params"]) == 0:
                rm_group_lst.append(id(group))

        self.param_groups = [
            g for g in self.param_groups if id(g) not in rm_group_lst  # type: ignore[has-type]
        ]

        # Add analog groups.
        for group in analog_param_groups:
            self.add_param_group(group)  # type: ignore[attr-defined]

    @no_grad()
    def step(self, closure: Optional[Callable] = None, **kwargs: Any) -> Optional[float]:
        """Perform an analog-aware single optimization step.

        If a group containing analog parameters is detected, the optimization
        step calls the related RPU controller. For regular parameter groups,
        the optimization step has the same behaviour as ``torch.optim.SGD``.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            kwargs: additional arguments if any

        Returns:
            The loss, if ``closure`` has been passed as a parameter.
        """
        # pylint: disable=too-many-branches
        # Update non-analog parameters using the given optimizer
        ret = super().step(closure, **kwargs)  # type: ignore[misc]

        # Update analog parameters
        for group in self.param_groups:
            learning_rate = group.get("lr")

            # Use analog_tile object.
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update.
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile

                    if analog_ctx.use_torch_update:
                        # In this case a separate weight parameter exists: do nothing.
                        continue

                    # Call `update` in the tile.
                    if not analog_ctx.has_gradient():
                        # Forward never used.
                        continue

                    # Update learning rate.
                    if learning_rate == 0.0:
                        analog_ctx.reset()
                        continue

                    if learning_rate is not None:
                        analog_tile.set_learning_rate(learning_rate)

                    runtime = analog_tile.get_runtime()
                    if analog_ctx.use_indexed:
                        for x_input, d_input in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update_indexed(
                                (
                                    x_input.to(analog_tile.device)
                                    if runtime.offload_input
                                    else x_input
                                ),
                                (
                                    d_input.to(analog_tile.device)
                                    if runtime.offload_gradient
                                    else d_input
                                ),
                            )
                    else:
                        x_input = cat(
                            analog_ctx.analog_input, axis=-1 if analog_tile.in_trans else 0
                        )
                        d_input = cat(
                            analog_ctx.analog_grad_output, axis=-1 if analog_tile.out_trans else 0
                        )
                        analog_tile.update(
                            x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                            d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                        )

                    analog_ctx.reset()

        # Apply post-update step operations (diffuse, decay, etc).
        # (only here because of unknown params order and shared weights)
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.post_update_step()
        return ret

    def set_learning_rate(self, learning_rate: float = 0.1) -> None:
        """Update the learning rate to a new value.

        Update the learning rate of the optimizer, propagating the changes
        to the analog tiles accordingly.

        Args:
            learning_rate: learning rate for the optimizer.
        """
        for param_group in self.param_groups:
            param_group["lr"] = learning_rate
            for param in param_group["params"]:
                if isinstance(param, AnalogContext):
                    # Update learning rate on the tile
                    param.analog_tile.set_learning_rate(learning_rate)


class AnalogMpOptimizerMixin(AnalogOptimizerMixin):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.freeze_analog_weight = False
        self.torch_step = None
        self.check_update_config()
        
    def zero_grad(self, *args: Any, **kwargs: Any):
        super().zero_grad(*args, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.reset() # remove the x and d signals

    def check_update_config(self):
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update.
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile
        
                    rpu_config = analog_tile.rpu_config

                    if isinstance(rpu_config.device, FloatingPointDevice):
                        continue

                    '''
                    Mixed precision update requires the tile are updated by
                    deterministic pulse update scheme, since the pulse number is
                    managed in the Python side.
                    '''
                    config_updated = False
                    
                    # if rpu_config.update.fixed_bl != True:
                    #     warnings.warn(f"rpu_config.update.fixed_bl was not set to True (current: {getattr(rpu_config.update, 'fixed_bl', 'not set')}). Setting to True.", UserWarning)
                    #     rpu_config.update.fixed_bl = True
                    #     config_updated = True

                    if rpu_config.update.pulse_type != PulseType.DETERMINISTIC_IMPLICIT:
                        warnings.warn(f"rpu_config.update.pulse_type was not set to PulseType.DETERMINISTIC_IMPLICIT (current: {getattr(rpu_config.update, 'pulse_type', 'not set')}). Setting to PulseType.DETERMINISTIC_IMPLICIT.", UserWarning)
                        rpu_config.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
                        config_updated = True
                    
                    # if rpu_config.update.update_bl_management != True:
                    #     warnings.warn(f"rpu_config.update.update_bl_management was not set to True (current: {getattr(rpu_config.update, 'update_bl_management', 'not set')}). Setting to True.", UserWarning)
                    #     rpu_config.update.update_bl_management = True
                    #     config_updated = True
                    
                    if rpu_config.update.update_management != True:
                        warnings.warn(f"rpu_config.update.update_management was not set to True (current: {getattr(rpu_config.update, 'update_management', 'not set')}). Setting to True.", UserWarning)
                        rpu_config.update_management = True
                        config_updated = True
                    
                    # Apply the configuration to analog_tile if any updates were made
                    if config_updated:
                        warnings.warn("RPU configuration was updated. Applying changes with analog_tile.to(real_rpu_config).", UserWarning)
                        analog_tile.to(rpu_config)
                        
    def get_processed_x_d_input(self, analog_ctx):
        analog_tile = analog_ctx.analog_tile
                
        runtime = analog_tile.get_runtime()
        if analog_ctx.use_indexed:
            for x_input, d_input in zip(
                analog_ctx.analog_input, analog_ctx.analog_grad_output
            ):
                if runtime.offload_input:
                    x_input = x_input.to(analog_tile.device)
                if runtime.offload_gradient:
                    d_input = d_input.to(analog_tile.device)

        else:
            x_input = cat(
                analog_ctx.analog_input, axis=-1 if analog_tile.in_trans else 0
            )
            d_input = cat(
                analog_ctx.analog_grad_output, axis=-1 if analog_tile.out_trans else 0
            )
                
            if runtime.offload_input:
                x_input = x_input.to(analog_tile.device)
            if runtime.offload_gradient:
                d_input = d_input.to(analog_tile.device)

        # deal with the mapping coefficient, etc.
        x_input, d_input = analog_tile.pre_update(
            x_input,
            0 if analog_tile.in_trans else x_input.dim() - 1,
            d_input,
            0 if analog_tile.out_trans else d_input.dim() - 1,
        )
                
        return x_input, d_input

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
        """Compute gradient using analog tile.

        Args:
            x_input: input activations
            d_input: gradient w.r.t. output activations

        Returns:
            The computed gradient.
        """
        # Compute gradient as average of outer products over batch
        # W: (N, M)
        # x_input: (B, M), d_input: (B, N)
        # gradient: (M, N) = sum(d_i ⊗ x_i) for i in batch
        # The avearage over the batchsize is implemented in the training code
        #   here we only sum up the gradient
        grad = torch.matmul(
            d_input.view(-1, d_input.size(-1)).transpose(0, 1),
            x_input.view(-1, x_input.size(-1))
        )
        
        return grad

    @no_grad()
    def prepare_grad(self) -> None:
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                
                # Call `update` in the tile.
                if not analog_ctx.has_gradient():
                    # Forward never used.
                    raise ValueError("what??")
                    return

                x_input, d_input = self.get_processed_x_d_input(analog_ctx)
                grad = self.compute_gradient(x_input, d_input)
                
                # TODO: use param as transfer buffer
                if param.size() != grad.size():
                    param.data = torch.zeros_like(grad)
                param.grad = grad
    
    @no_grad()
    def step(self, closure: Optional[Callable] = None, **kwargs: Any) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # pylint: disable=too-many-branches
        self.prepare_grad()
        
        # ====================================
        # Update non-analog parameters using the given optimizer
        # call torch optimizer and achieve the desired update on AdamW, shored in param
        # ====================================
        assert self.torch_step is not None, "need to define torch_step"
        ret = self.torch_step(self, closure, **kwargs)  # type: ignore[misc]
        
        if self.freeze_analog_weight:
            return ret

        # ====================================
        # analog update: transfer
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update.
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile

                    if analog_ctx.use_torch_update:
                        # In this case a separate weight parameter exists: do nothing.
                        continue

                    # # Call `update` in the tile.
                    # if not analog_ctx.has_gradient():
                    #     # Forward never used.
                    #     continue

                    analog_tile.set_learning_rate(1.)

                    rpu_device = self._get_RPU_device(analog_tile.rpu_config)
                    if not isinstance(rpu_device, FloatingPointDevice):
                        # Don't need to scale dw_min by mapping coefficient
                        # The mapping coefficient is already considered in the gradient computation
                        dw_min = rpu_device.dw_min
                        max_bit_length:int = analog_tile.rpu_config.update.desired_bl
                    else:
                        dw_min = max_bit_length = None
                        
                    runtime = analog_tile.get_runtime()
                    if analog_ctx.use_indexed:
                        raise NotImplementedError(
                            "Indexed updates are not supported in MP."
                        )
                        for x_input, d_input in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update_indexed(
                                (
                                    x_input.to(analog_tile.device)
                                    if runtime.offload_input
                                    else x_input
                                ),
                                (
                                    d_input.to(analog_tile.device)
                                    if runtime.offload_gradient
                                    else d_input
                                ),
                            )
                    else:
                        if isinstance(rpu_device, FloatingPointDevice):
                            desired_update = param
                            analog_tile.update(
                                torch.eye(analog_tile.in_size).to(analog_tile.device), 
                                -desired_update.transpose(0, 1)
                            )
                        else:
                            desired_pulse = param.div(dw_min).trunc()
                            available_pulse = desired_pulse.clamp(
                                min=-max_bit_length, max=max_bit_length)
                            desired_update = available_pulse.mul(dw_min)

                        # analog_tile.update(
                        #     torch.eye(analog_tile.in_size).to(analog_tile.device), 
                        #     -desired_update.transpose(0, 1)
                        # )
                        self.update_tile(analog_tile, desired_update)
                        if hasattr(analog_ctx, 'writable'):
                            with analog_ctx.writable():
                                param.data.sub_(desired_update)
                        else:
                            param.data.sub_(desired_update)
                    
        # Apply post-update step operations (diffuse, decay, etc).
        # (only here because of unknown params order and shared weights)
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.post_update_step()
                    
        return ret
    
    @staticmethod
    def update_tile(tile, desired_update):
        '''Define a function here since the subclass may need to update other tiles.'''
        if not hasattr(tile, '_eye_matrix'):
            tile._eye_matrix = torch.eye(tile.in_size, device=tile.device)
        tile.update(
            tile._eye_matrix,
            -desired_update.transpose(0, 1)
        )


class AnalogGradMpOptimizerMixin(AnalogMpOptimizerMixin):
    def __init__(self, params, 
                 grad_rpu_config=None, scaling_grad=True, 
                 reset_grad=True, 
                 grad_tile_n_state=None, grad_desired_bl=255,
                 split_signal=False, 
                 n_grad_sample=1,
                #  shuffle_grad_columns=True,
                 shuffle_grad_columns=False,
                 grad_use_diffx=False,
                 adaptive_peak_fallback=False,
                 adaptive_peak_ratio_threshold=2.0,
                 adaptive_fallback_use_diffx=True,
                 # Kept for backward compatibility; currently no-op.
                 split_signal_reverse_groups=False,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(params, *args, **kwargs)
        self.torch_step = None
        self.grad_rpu_config = grad_rpu_config
        self.scaling_grad = scaling_grad
        self.reset_grad = reset_grad
        self.grad_tile_n_state = grad_tile_n_state
        self.grad_desired_bl = grad_desired_bl
        self.analog_state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
        self._dw_min = None  # set in _get_grad_RPU_tile
        
        self.split_signal = split_signal
        self.split_K = 32
        self.n_grad_sample = n_grad_sample
        self.shuffle_grad_columns = shuffle_grad_columns
        # New feature flags:
        # - grad_use_diffx: use (+x, d) and (-x, d) differential updates.
        # - adaptive_peak_fallback: detect destructive cancellation in split-signal path.
        # - adaptive_peak_ratio_threshold: trigger threshold for fallback.
        # - adaptive_fallback_use_diffx: whether no-split fallback also uses diffx.
        self.grad_use_diffx = grad_use_diffx
        self.adaptive_peak_fallback = adaptive_peak_fallback
        self.adaptive_peak_ratio_threshold = adaptive_peak_ratio_threshold
        self.adaptive_fallback_use_diffx = adaptive_fallback_use_diffx
        # Cached debug/runtime signals from the latest prepare_grad pass.
        self._last_peak_ratio = None
        self._last_used_adaptive_fallback = False

        def rr_func_(X):
            condition = (X >= 0)
            true_values = -torch.log(1.0 + 1e-5 - X)
            false_values = torch.log(1.0 + 1e-5 + X)
            result = torch.where(condition, true_values, false_values)
            return result
        self.response_recover = torch.vmap(rr_func_)
        
        self._init_analog_grad_buffer()

    def _init_analog_grad_buffer(self):
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    state = self.analog_state[param]
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile
                    grad_tile = self._get_grad_RPU_tile(analog_tile)
                    state["grad_tile"] = grad_tile
                    state["grad_tile_pool"] = [self._get_grad_RPU_tile(analog_tile) for _ in range(10)]
    
    def _get_grad_RPU_tile(self, analog_tile):
        """
        Create and configure a gradient RPU tile based on the provided analog tile.
        This method generates a gradient computation tile with RPU (Resistive Processing Unit)
        configuration derived from the input analog tile. It handles configuration inheritance,
        device-specific settings, and applies gradient-specific constraints.
        Args:
            analog_tile: The analog tile whose configuration serves as the basis for the
                        gradient tile. Must have attributes: rpu_config, in_size, out_size.
        Returns:
            An initialized analog tile module configured for gradient computation with
            dimensions (out_features, in_features).
        Notes:
            - If self.grad_rpu_config is not provided, a new config is created by copying
              attributes from the input analog tile's rpu_config.
            - For non-floating point devices, dw_min and desired_bl are set from
              grad_tile_n_state and grad_desired_bl respectively.
            - Warnings are issued if gradient-specific parameters are provided but
              ignored (e.g., when using FloatingPointDevice).
            - Weight scaling and output scaling learning are disabled for gradient tiles
              to prevent mapping-based scaling.
            - Sets self._dw_min attribute from the gradient device if available.
        """
        
        rpu_config_weight = analog_tile.rpu_config
        in_features = analog_tile.in_size
        out_features = analog_tile.out_size

        if self.grad_rpu_config is not None:
            rpu_config_grad = self.grad_rpu_config
        else:
            rpu_device_grad = self._get_RPU_device(rpu_config_weight)
            rpu_config_grad = SingleRPUConfig(device=rpu_device_grad)

            # Copy other attributes
            for attr in dir(rpu_config_weight):
                if attr in ['device']:
                    continue
                if not attr.startswith("__") and hasattr(rpu_config_grad, attr):
                    setattr(rpu_config_grad, attr, getattr(rpu_config_weight, attr))
                
            # set dw_min and desired_bl if not floating point device
            if not isinstance(rpu_config_grad.device, FloatingPointDevice):
                if self.grad_tile_n_state is not None:
                    rpu_config_grad.device.dw_min = 2 / self.grad_tile_n_state
                if self.grad_desired_bl is not None:
                    rpu_config_grad.update.desired_bl = self.grad_desired_bl
            else:
                if self.grad_tile_n_state is not None:
                    warnings.warn(
                        "grad_tile_n_state was provided but is ignored when using a FloatingPointDevice for the gradient tile. "
                        f"(grad_tile_n_state={self.grad_tile_n_state})",
                        UserWarning,
                    )
                if self.grad_desired_bl is not None:
                    warnings.warn(
                        "grad_desired_bl was provided but is ignored when using a FloatingPointDevice for the gradient tile. "
                        f"(grad_desired_bl={self.grad_desired_bl})",
                        UserWarning,
                    )
        
            # Gradient tile should not be scaled by mapping
            rpu_config_grad.mapping.weight_scaling_omega = 0.0
            rpu_config_grad.mapping.learn_out_scaling = False
            
            # Weight RPU may use deterministic pulse, 
            #   while gradient RPU should use stochastic pulse
            if hasattr(rpu_config_grad.update, "pulse_type"):
                rpu_config_grad.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
            if hasattr(rpu_config_grad.update, "update_management"):
                rpu_config_grad.update.update_management = True
            if hasattr(rpu_config_grad.update, "update_bl_management"):
                rpu_config_grad.update.update_bl_management = True

        if hasattr(rpu_config_grad.device, 'dw_min'):
            self._dw_min = rpu_config_grad.device.dw_min
        else:
            self._dw_min = None
        
        tile_module_class = rpu_config_grad.get_default_tile_module_class(out_features, in_features)
        analog_tile_grad = tile_module_class(out_features, in_features, rpu_config_grad, False)
        analog_tile_grad = analog_tile_grad.to(analog_tile.device)
        # Rebind shared weight tensor after device migration so that
        # _get_tile_weights_ref() returns a tensor on the correct device.
        if hasattr(analog_tile_grad, '_bind_shared_weights'):
            analog_tile_grad._shared_weight_tensor = None
            analog_tile_grad._bind_shared_weights()
        elif not hasattr(analog_tile_grad, '_get_tile_weights_ref'):
            # Fallback: set up shared weights manually.
            # CUDA tile expects (in, out); CPU tile expects (out, in).
            if "Cuda" in type(analog_tile_grad.tile).__name__:
                sw = torch.zeros(in_features, out_features, device=analog_tile.device)
            else:
                sw = torch.zeros(out_features, in_features, device=analog_tile.device)
            analog_tile_grad.tile.set_shared_weights(sw)
            analog_tile_grad._shared_weight_tensor = sw
        # Ensure gradient accumulation is not scaled by the tile LR.
        analog_tile_grad.set_learning_rate(1.0)

        self._zero_tile(analog_tile_grad)

        return analog_tile_grad

    def _get_tile_weights_ref(self, analog_tile):
        """Get a GPU-resident reference to tile weights in (out, in) layout."""
        if hasattr(analog_tile, '_get_tile_weights_ref'):
            return analog_tile._get_tile_weights_ref()
        # Fallback: shared weight tensor layout differs by tile flavor.
        # CUDA: sw is (in, out) — transpose to get (out, in).
        # CPU:  sw is (out, in) — return as-is.
        if hasattr(analog_tile, '_shared_weight_tensor') and analog_tile._shared_weight_tensor is not None:
            if "Cuda" in type(analog_tile.tile).__name__:
                return analog_tile._shared_weight_tensor.t()
            return analog_tile._shared_weight_tensor
        # Last resort: copy from tile (slow, CPU roundtrip)
        w, _ = analog_tile.get_weights()
        return w.to(analog_tile.device)

    def _zero_tile(self, analog_tile):
        ref = self._get_tile_weights_ref(analog_tile)
        ref.zero_()

    def zero_grad(self, *args: Any, **kwargs: Any):
        super().zero_grad(*args, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    analog_ctx = param
                    analog_state = self.analog_state[param]
                    grad_tile = analog_state["grad_tile"]
                    if self.reset_grad:
                        self._zero_tile(grad_tile)
                    analog_ctx.reset()
                    
    def _debug_prepare_true_grad(self):
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                
                # Call `update` in the tile.
                if not analog_ctx.has_gradient():
                    # Forward never used.
                    raise ValueError("what??")
                    return

                x_input, d_input = self.get_processed_x_d_input(analog_ctx)
                grad = self.compute_gradient(x_input, d_input)
                
                # TODO: use param as transfer buffer
                if param.size() != grad.size():
                    param.data = torch.zeros_like(grad)
                param.true_grad = grad

    @no_grad()
    def _get_grad_by_rank_update(
        self,
        grad_tile,
        x_input,
        d_input,
        shuffle_columns: Optional[bool] = None,
        use_diffx: Optional[bool] = None,
    ):
        self._zero_tile(grad_tile)
        do_col_shuffle = self.shuffle_grad_columns if shuffle_columns is None else shuffle_columns
        do_diffx = self.grad_use_diffx if use_diffx is None else use_diffx

        if self.scaling_grad:
            # average over coordinates
            x_mean, x_max = x_input.mean(), x_input.abs().max()
            d_mean, d_max = d_input.mean(), d_input.abs().max()
            # sum over batch
            x_sum = x_input.sum(dim=tuple(range(x_input.ndim - 1)))
            d_sum = d_input.sum(dim=tuple(range(d_input.ndim - 1)))
            
            batch_size = x_input.numel() // x_input.size(-1)

            # coordinate-wise normalization
            small_val = 1e-11
            x_input_norm = (x_input - x_mean) / (x_max + small_val)
            d_input_norm = (d_input - d_mean) / (d_max + small_val)

            if self._dw_min is not None:
                desired_bl = self.grad_desired_bl
                dw_min = self._dw_min
                C = (desired_bl * dw_min)**(1/2)
                # C = (desired_bl * dw_min)**(1/2) * 1.5
            else:
                C = 1.0

            x_update = C * x_input_norm
            d_update = C * d_input_norm
        else:
            # no normalization
            x_update = x_input
            d_update = d_input

        if do_col_shuffle:
            x_col_perm = torch.randperm(x_update.shape[-1], device=x_update.device)
            d_col_perm = torch.randperm(d_update.shape[-1], device=d_update.device)
            x_update = x_update.index_select(-1, x_col_perm)
            d_update = d_update.index_select(-1, d_col_perm)

        if do_diffx:
            # Differential update with +/-x reduces systematic update bias.
            grad_tile.update(x_update, d_update)
            grad_pre_pos = self._get_tile_weights_ref(grad_tile).neg()

            self._zero_tile(grad_tile)
            grad_tile.update(-x_update, d_update)
            grad_pre_neg = self._get_tile_weights_ref(grad_tile).neg()
            grad = (grad_pre_pos - grad_pre_neg) / 2.0
        else:
            grad_tile.update(x_update, d_update)
            # TODO: ideal reading
            grad = self._get_tile_weights_ref(grad_tile).neg()

        if do_col_shuffle:
            inv_x_col_perm = torch.empty_like(x_col_perm)
            inv_x_col_perm[x_col_perm] = torch.arange(x_col_perm.numel(), device=x_col_perm.device)
            inv_d_col_perm = torch.empty_like(d_col_perm)
            inv_d_col_perm[d_col_perm] = torch.arange(d_col_perm.numel(), device=d_col_perm.device)
            grad = grad.index_select(0, inv_d_col_perm).index_select(1, inv_x_col_perm)
        
        # Response recovery, which will apply for non-floating point device
        # if self._dw_min is not None:
        #     grad = self.response_recover(grad)

        if self.scaling_grad:
            # recover gradient
            grad.div_(C**2)
            grad.mul_(x_max + small_val).mul_(d_max + small_val)
            grad.add_(x_sum.unsqueeze(0), alpha=d_mean)
            grad.add_(d_sum.unsqueeze(1), alpha=x_mean)
            grad.sub_(batch_size * x_mean * d_mean)
                
        return grad
    
    @no_grad()
    def _get_split_signal_indices(self, x_input, d_input):
        # Flatten batch for easier indexing
        # x_flat = x_input.view(-1, x_input.size(-1))
        d_flat = d_input.view(-1, d_input.size(-1))
        
        # Compute grad_magnitude for each batch: Ni = ||x_i|| * ||d_i||
        # x_norms = x_flat.norm(dim=-1)
        d_norms = d_flat.norm(dim=-1)
        # grad_magnitude = x_norms * d_norms
        grad_magnitude = d_norms
        # Grouping parameters
        # gamma = 0.1
        gamma = 0.5
        min_grad = 1e-15
        # gamma = self.split_K / self.DRU_BL
        _min_val, _max_val = torch.aminmax(grad_magnitude)
        max_grad_norm = _max_val.item()
        min_grad_norm = _min_val.item()
        # Compute delimiters for grouping
        delimiters = [max_grad_norm]
        while delimiters[-1] > min_grad_norm and delimiters[-1] > min_grad:
            delimiters.append(delimiters[-1] * gamma)
        delimiters.append(0)  # Ensure last group includes min
            
        # Assign indices to groups
        MAX_BIN_SIZE = 500
        group_indices = []
        for i in range(len(delimiters)-1):
            group_mask = (grad_magnitude <= delimiters[i]) & (grad_magnitude > delimiters[i+1])
            indices = torch.where(group_mask)[0]
            if indices.numel() > MAX_BIN_SIZE:
                # Split indices into subgroups of size 1000
                num_subgroups = (indices.numel() + MAX_BIN_SIZE - 1) // MAX_BIN_SIZE
                for j in range(num_subgroups):
                    start = j * MAX_BIN_SIZE
                    end = min((j + 1) * MAX_BIN_SIZE, indices.numel())
                    group_indices.append(indices[start:end])
            elif indices.numel() > 0:
                group_indices.append(indices)
    
        return group_indices    
    
    @no_grad()
    def _get_grad_by_rank_update_split_signal(
        self,
        grad_tile,
        x_input,
        d_input,
        shuffle_columns: Optional[bool] = None,
        use_diffx: Optional[bool] = None,
        return_peak_ratio: bool = False,
    ):
        """Compute split-signal gradient and optionally report peak-ratio stats.

        peak_ratio is defined as max(||acc_grad_t||) / ||final_acc_grad|| over the
        accumulation process. A large value indicates cancellation after an earlier
        peak and can be used to trigger no-split fallback.
        """
        # Flatten batch for easier indexing
        x_flat = x_input.view(-1, x_input.size(-1))
        d_flat = d_input.view(-1, d_input.size(-1))
        
        group_indices = self._get_split_signal_indices(x_input, d_input)

        # =====================================================================
        # Use zeros_like(ref) so the accumulator has the same memory strides as the
        # shared weight tensor (.t() view for CUDA tiles). This ensures param.grad and
        # param.data have matching layouts, which is required by fused AdamW.
        grad = torch.zeros_like(self._get_tile_weights_ref(grad_tile))
        max_acc_norm = 0.0
        sum_group_norms = 0.0
        for idx, indices in enumerate(group_indices):
            if indices.numel() == 0:
                continue
            xg = x_flat[indices]
            dg = d_flat[indices]
            
            analog_grad = self._get_grad_by_rank_update(
                grad_tile,
                xg,
                dg,
                shuffle_columns=shuffle_columns,
                use_diffx=use_diffx,
            )
            if analog_grad.isnan().any():
                pass
            
            grad.add_(analog_grad)
            if return_peak_ratio:
                sum_group_norms += analog_grad.norm().item()
                cur_norm = grad.norm().item()
                if cur_norm > max_acc_norm:
                    max_acc_norm = cur_norm
        
        if return_peak_ratio:
            final_norm = grad.norm().item()
            peak_ratio = max_acc_norm / (final_norm + 1e-8)
            return grad, peak_ratio, sum_group_norms
            
        return grad
    
    @no_grad()
    def _get_grad_by_rank_update_wrapper(self, analog_tile, x_flat, d_flat, subtask_input_idx):
        # grad_tile = self._get_grad_RPU_tile(analog_tile)
        
        #TODO
        grad_tile = analog_tile
        self.zero_tile(grad_tile)
                
        if subtask_input_idx is not None:
            analog_grad = self._get_grad_by_rank_update(
                grad_tile, 
                x_flat[subtask_input_idx],
                d_flat[subtask_input_idx]
            )
        else:
            # Use full flattened tensors for non-split mode
            analog_grad = self._get_grad_by_rank_update(
                grad_tile, x_flat, d_flat
            )
        return analog_grad
    
    @no_grad()
    def prepare_grad(self) -> None:
        if self.freeze_analog_weight:
            return
        # self.prepare_grad_parallel()
        self.prepare_grad_old()
    
    @no_grad()
    def prepare_grad_parallel(self) -> None:
        '''Redefine the function. Prepare gradient using analog tile.'''
        
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param

                # state = self.analog_state[param]
                # grad_tile = state["grad_tile"]
                
                # Call `update` in the tile.
                if not analog_ctx.has_gradient():
                    # Forward never used.
                    raise ValueError("There is no gradient. Run the backward pass first.")
                
                x_input, d_input = self.get_processed_x_d_input(analog_ctx)
                
                # Flatten inputs for consistent indexing
                x_flat = x_input.view(-1, x_input.size(-1))
                d_flat = d_input.view(-1, d_input.size(-1))
                
                if self.split_signal:
                    group_indices = self._get_split_signal_indices(x_input, d_input)
                    # Repeat group indices for n_grad_sample times
                    subtask_input_indices = group_indices * self.n_grad_sample
                else:
                    # For non-split mode, use None as placeholder
                    subtask_input_indices = [None for _ in range(self.n_grad_sample)]
                
                n_streams = len(subtask_input_indices)
                cuda_streams = [
                    torch.cuda.Stream() for _ in range(n_streams)
                ]
                
                analog_grad_list = [None for _ in range(n_streams)]
                analog_tile = analog_ctx.analog_tile
                
                while len(self.analog_state[param]["grad_tile_pool"]) < n_streams:
                    self.analog_state[param]["grad_tile_pool"].append(
                        self._get_grad_RPU_tile(analog_tile)
                    )
                
                for s_idx, subtask_input_idx in enumerate(subtask_input_indices):
                    with torch.cuda.stream(cuda_streams[s_idx]):
                        analog_grad_list[s_idx] = self._get_grad_by_rank_update_wrapper(
                            # analog_tile, 
                            self.analog_state[param]["grad_tile_pool"][s_idx],
                            x_flat, d_flat, subtask_input_idx
                        )
                
                # Synchronize all streams
                for stream in cuda_streams:
                    stream.synchronize()
                
                # Verify all gradients computed successfully
                if any(g is None for g in analog_grad_list):
                    failed_indices = [i for i, g in enumerate(analog_grad_list) if g is None]
                    raise RuntimeError(f"Some gradients failed to compute at indices: {failed_indices}")
                
                grad = torch.stack(analog_grad_list, dim=0).sum(dim=0)
                grad.div_(self.n_grad_sample)
                
                # =======================================
                # Sample gradient n_grad_sample times and take the average
                # grad_sum = None
                # for sample_idx in range(self.n_grad_sample):
                #     # =======================================
                #     # TODO: re-initialize the grad tile state
                #     self._init_analog_grad_buffer()
                #     # =======================================
                #     if self.split_signal:
                #         grad_sample = self._get_grad_by_rank_update_split_signal(grad_tile, x_input, d_input)
                #     else:
                #         grad_sample = self._get_grad_by_rank_update(grad_tile, x_input, d_input)
                    
                #     if grad_sum is None:
                #         grad_sum = grad_sample
                #     else:
                #         grad_sum.add_(grad_sample)
                # =======================================
                
                # Average the gradient samples
                # grad = grad_sum.div_(self.n_grad_sample)
                    
                # TODO: use param as transfer buffer
                if param.size() != grad.size():
                    param.data = torch.zeros_like(grad)
                    
                # =========================================
                # true_grad = torch.matmul(
                #     d_input.view(-1, d_input.size(-1)).transpose(0, 1),
                #     x_input.view(-1, x_input.size(-1))
                # )
                # sign_mask = torch.sign(grad) != torch.sign(true_grad)
                # grad[sign_mask] = true_grad[sign_mask]
                # =========================================
                
                
            
                param.grad = grad
                
                # =========================================
                # if random.random() < 1:
                if random.random() < 0.02:
                # if random.random() < 0.3:
                    true_grad = torch.matmul(
                        d_input.view(-1, d_input.size(-1)).transpose(0, 1),
                        x_input.view(-1, x_input.size(-1))
                    )
                
                    abs_err = (true_grad - grad).norm()
                    rel_err = abs_err / (true_grad.norm() + 1e-8)
                    # magnitude_norm = grad.abs().mean()
                    # magnitude_true = true_grad.abs().mean()
                    cosine = cosine_similarity(true_grad.flatten(), grad.flatten(), dim=0)
                    sign_agreement_n = (torch.sign(grad) == torch.sign(true_grad)).sum().item()
                    vector_size = grad.numel()
                    sign_agreement = (torch.sign(grad) == torch.sign(true_grad)).float().mean().item()
                    element_wise_ratio = grad / (true_grad + true_grad.sign()*1e-8)

                    element_wise_ratio = grad / (true_grad + true_grad.sign()*1e-8)
                    max_ratio = element_wise_ratio.max().item()
                    min_ratio = element_wise_ratio.min().item()
                    min_abs_ratio = element_wise_ratio.abs().min().item()
                    max_abs_ratio = element_wise_ratio.abs().max().item()

                    print(f"relative error: {rel_err.item() * 100:.3f}%, cosine: {cosine.item():.6f}, norm ratio: {(grad.norm() / (true_grad.norm() + 1e-8)).item():.6f}, sign agreement: {sign_agreement*100:.2f}% ({sign_agreement_n}/{vector_size}), ratio: [{min_ratio:.3f}, {max_ratio:.3f}], abs_ratio: [{min_abs_ratio:.6f}, {max_abs_ratio:.6f}]")
                    # =========================================
                    # if max_abs_ratio > 100:
                    #     import os
                    #     import matplotlib.pyplot as plt
                    #     plt.figure(figsize=(10, 6))
                    #     plt.scatter(true_grad.detach().cpu().flatten().numpy(), 
                    #                 element_wise_ratio.detach().cpu().flatten().numpy(), 
                    #                 alpha=0.5, s=10)
                    #     plt.xlabel('True Gradient')
                    #     plt.ylabel('Element-wise Ratio (grad / true_grad)')
                    #     plt.title('Gradient vs Element-wise Ratio')
                    #     plt.grid(True, alpha=0.3)
                    #     plt.tight_layout()
                    #     plt.yscale('symlog', linthresh=1e-6)
                    #     plt.xscale('symlog', linthresh=1e-6)

                    #     save_dir = os.path.join(os.path.dirname(__file__), 'report', 'cache')
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = os.path.join(save_dir, 'grad_ratio_scatter.png')
                    #     plt.savefig(save_path, dpi=150)
                    #     plt.close()

                    #     print(f"save to: {save_path}")
                    #     max_ratio = element_wise_ratio.max().item()
                    #     min_ratio = element_wise_ratio.min().item()
                    #     min_abs_ratio = element_wise_ratio.abs().min().item()
                    #     max_abs_ratio = element_wise_ratio.abs().max().item()
                    #     raise SystemExit("stop here for debug")
                    # =========================================

                        # if self.split_signal:
                        #     print(f"Grouping into {len(delimiters)-1} groups with delimiters: grad_magnitude: max={grad_magnitude.max().item():.6f}, min={grad_magnitude.min().item():.6f}")
                        # for i, indices in enumerate(group_indices):
                        #     print(f"  Group {i:2d}: {indices.numel():5d} samples, delimiter=({delimiters[i]:.3e}, {delimiters[i+1]:.3e})")
                        
                    # if rel_err > 1:
                    #     # Pass computed values to avoid recomputation
                    #     layer_data = {
                    #         'x_input': x_input,
                    #         'd_input': d_input,
                    #         'grad': grad,
                    #         'true_grad': true_grad,
                    #         'abs_err': abs_err.item(),
                    #         'rel_err': rel_err.item(),
                    #         'cosine_sim': cosine.item()
                    #     }
                    #     self.save_grad(layer_data_dict=layer_data)
                    #     raise SystemExit("Rel error too large!")
    
    @no_grad()
    def prepare_grad_old(self) -> None:
        '''Redefine the function. Prepare gradient using analog tile.'''
        
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param

                state = self.analog_state[param]
                grad_tile = state["grad_tile"]
                
                # Call `update` in the tile.
                if not analog_ctx.has_gradient():
                    # Forward never used.
                    raise ValueError("There is no gradient. Run the backward pass first.")
                
                x_input, d_input = self.get_processed_x_d_input(analog_ctx)
                
                # =======================================
                # Sample gradient n_grad_sample times and take the average
                grad_sum = None
                last_peak_ratio = None
                used_adaptive_fallback = False
                for sample_idx in range(self.n_grad_sample):
                    # =======================================
                    # TODO: re-initialize the grad tile state
                    # self._init_analog_grad_buffer()
                    # =======================================
                    if self.split_signal:
                        if self.adaptive_peak_fallback:
                            # First run split-signal path and measure cancellation.
                            grad_sample, peak_ratio, _sum_group_norms = self._get_grad_by_rank_update_split_signal(
                                grad_tile,
                                x_input,
                                d_input,
                                shuffle_columns=self.shuffle_grad_columns,
                                use_diffx=self.grad_use_diffx,
                                return_peak_ratio=True,
                            )
                            last_peak_ratio = peak_ratio
                            if peak_ratio > self.adaptive_peak_ratio_threshold:
                                # Fallback path: recompute with no-split strategy.
                                grad_sample = self._get_grad_by_rank_update(
                                    grad_tile,
                                    x_input,
                                    d_input,
                                    shuffle_columns=self.shuffle_grad_columns,
                                    use_diffx=self.adaptive_fallback_use_diffx,
                                )
                                used_adaptive_fallback = True
                        else:
                            grad_sample = self._get_grad_by_rank_update_split_signal(
                                grad_tile,
                                x_input,
                                d_input,
                                shuffle_columns=self.shuffle_grad_columns,
                                use_diffx=self.grad_use_diffx,
                            )
                    else:
                        grad_sample = self._get_grad_by_rank_update(
                            grad_tile,
                            x_input,
                            d_input,
                            shuffle_columns=self.shuffle_grad_columns,
                            use_diffx=self.grad_use_diffx,
                        )
                    
                    if grad_sum is None:
                        grad_sum = grad_sample
                    else:
                        grad_sum.add_(grad_sample)
                        
                # =======================================
                # Average the gradient samples
                grad = grad_sum.div_(self.n_grad_sample)
                self._last_peak_ratio = last_peak_ratio
                self._last_used_adaptive_fallback = used_adaptive_fallback
                    
                # TODO: use param as transfer buffer
                if param.size() != grad.size():
                    param.data = torch.zeros_like(grad)
                    
                # =========================================
                # true_grad = torch.matmul(
                #     d_input.view(-1, d_input.size(-1)).transpose(0, 1),
                #     x_input.view(-1, x_input.size(-1))
                # )
                # sign_mask = torch.sign(grad) != torch.sign(true_grad)
                # grad[sign_mask] = true_grad[sign_mask]
                # =========================================
                    
                # =========================================
                # # 挑出grad中绝对值最大的30%的元素，替换成对应位置的true_grad元素
                # true_grad = torch.matmul(
                #     d_input.view(-1, d_input.size(-1)).transpose(0, 1),
                #     x_input.view(-1, x_input.size(-1))
                # )
                # grad_abs = grad.abs()
                # threshold = torch.quantile(grad_abs, 0.7)  # 30%最大值对应70%的分位数
                # mask = grad_abs >= threshold
                # grad[mask] = true_grad[mask]
                # =========================================
                    
                # =========================================
                # true_grad = torch.matmul(
                #     d_input.view(-1, d_input.size(-1)).transpose(0, 1),
                #     x_input.view(-1, x_input.size(-1))
                # )
                # # Sort grad by absolute value and reorder true_grad accordingly
                # Digital Element + Analog Order
                # grad_abs = grad.abs()
                # sorted_indices = torch.argsort(grad_abs.flatten())
                # grad_sorted = grad.flatten()[sorted_indices]
                # true_grad_reordered = true_grad.flatten()[sorted_indices]
                # grad = true_grad_reordered.reshape(true_grad.shape)
                # 
                # Analog Element + Digital Order
                # true_grad_abs = true_grad.abs()
                # sorted_indices = torch.argsort(true_grad_abs.flatten())
                # true_grad_sorted = true_grad.flatten()[sorted_indices]
                # grad_reordered = grad.flatten()[sorted_indices]
                # grad = grad_reordered.reshape(true_grad.shape)
                # =========================================

                param.grad = grad
                # Snapshot the raw analog gradient before any optimizer (e.g. Muon's
                # muon_update via grad.lerp_) can mutate param.grad in-place. Logging
                # code that wants the un-mutated analog grad reads this attribute.
                param._raw_analog_grad = grad.detach().clone()

                # =========================================
                # if random.random() < 1:
                if random.random() < 0.02:
                # if random.random() < 0.3:
                    true_grad = torch.matmul(
                        d_input.view(-1, d_input.size(-1)).transpose(0, 1),
                        x_input.view(-1, x_input.size(-1))
                    )
                
                    abs_err = (true_grad - grad).norm()
                    rel_err = abs_err / (true_grad.norm() + 1e-8)
                    # magnitude_norm = grad.abs().mean()
                    # magnitude_true = true_grad.abs().mean()
                    cosine = cosine_similarity(true_grad.flatten(), grad.flatten(), dim=0)
                    sign_agreement_n = (torch.sign(grad) == torch.sign(true_grad)).sum().item()
                    vector_size = grad.numel()
                    sign_agreement = (torch.sign(grad) == torch.sign(true_grad)).float().mean().item()
                    element_wise_ratio = grad / (true_grad + true_grad.sign()*1e-8)

                    element_wise_ratio = grad / (true_grad + true_grad.sign()*1e-8)
                    max_ratio = element_wise_ratio.max().item()
                    min_ratio = element_wise_ratio.min().item()
                    min_abs_ratio = element_wise_ratio.abs().min().item()
                    max_abs_ratio = element_wise_ratio.abs().max().item()

                    # Kendall's tau between flattened true_grad and grad.
                    true_flat = true_grad.flatten()
                    grad_flat = grad.flatten()
                    tau_sample_size = 2048
                    if true_flat.numel() > tau_sample_size:
                        sample_idx = torch.randperm(true_flat.numel(), device=true_flat.device)[:tau_sample_size]
                        true_flat_tau = true_flat.index_select(0, sample_idx)
                        grad_flat_tau = grad_flat.index_select(0, sample_idx)
                    else:
                        true_flat_tau = true_flat
                        grad_flat_tau = grad_flat

                    from scipy.stats import kendalltau  # type: ignore
                    kendall_tau = kendalltau(
                        true_flat_tau.detach().cpu().numpy(),
                        grad_flat_tau.detach().cpu().numpy(),
                        nan_policy="omit",
                    ).statistic

                    print(f"Error: {rel_err.item() * 100:.3f}%, Cos: {cosine.item():.6f}, Kendall tau: {kendall_tau:.6f}, norm ratio: {(grad.norm() / (true_grad.norm() + 1e-8)).item():.6f}, sign agreement: {sign_agreement*100:.2f}% ({sign_agreement_n}/{vector_size}), ratio: [{min_ratio:.3f}, {max_ratio:.3f}], abs_ratio: [{min_abs_ratio:.6f}, {max_abs_ratio:.6f}]")

                    # =========================================
                    # if max_abs_ratio > 100:
                    #     import os
                    #     import matplotlib.pyplot as plt
                    #     plt.figure(figsize=(10, 6))
                    #     plt.scatter(true_grad.detach().cpu().flatten().numpy(), 
                    #                 element_wise_ratio.detach().cpu().flatten().numpy(), 
                    #                 alpha=0.5, s=10)
                    #     plt.xlabel('True Gradient')
                    #     plt.ylabel('Element-wise Ratio (grad / true_grad)')
                    #     plt.title('Gradient vs Element-wise Ratio')
                    #     plt.grid(True, alpha=0.3)
                    #     plt.tight_layout()
                    #     plt.yscale('symlog', linthresh=1e-6)
                    #     plt.xscale('symlog', linthresh=1e-6)

                    #     save_dir = os.path.join(os.path.dirname(__file__), 'report', 'cache')
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = os.path.join(save_dir, 'grad_ratio_scatter.png')
                    #     plt.savefig(save_path, dpi=150)
                    #     plt.close()

                    #     print(f"save to: {save_path}")
                    #     max_ratio = element_wise_ratio.max().item()
                    #     min_ratio = element_wise_ratio.min().item()
                    #     min_abs_ratio = element_wise_ratio.abs().min().item()
                    #     max_abs_ratio = element_wise_ratio.abs().max().item()
                    #     raise SystemExit("stop here for debug")
                    # =========================================

                        # if self.split_signal:
                        #     print(f"Grouping into {len(delimiters)-1} groups with delimiters: grad_magnitude: max={grad_magnitude.max().item():.6f}, min={grad_magnitude.min().item():.6f}")
                        # for i, indices in enumerate(group_indices):
                        #     print(f"  Group {i:2d}: {indices.numel():5d} samples, delimiter=({delimiters[i]:.3e}, {delimiters[i+1]:.3e})")
                        
                    # if rel_err > 0.5:
                    #     self.save_grad()
                    #     raise SystemExit("Rel error too large!")
         
    def save_grad(self, epoch=None):
        """
        Save gradient data and analysis results.
        """
        # Cache and save x_input, d_input, and grad for each AnalogContext layer
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        signal_dir = os.path.join(
            os.path.dirname(__file__), '..', 'cache',
            f'A01-signal-epoch{epoch if epoch is not None else "NA"}-{timestamp}'
        )
        os.makedirs(signal_dir, exist_ok=True)
        tensor_info = []
        param_names = []
        error_stats = []

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group["params"]):
                if not isinstance(param, AnalogContext):
                    continue
                analog_ctx = param
                # Generate a unique name for the parameter
                param_name_map = getattr(self, "_param_name_by_id", {})
                raw_name = param_name_map.get(id(param), None)
                if raw_name is None:
                    name = f'group{group_idx}_param{id(param)}'
                else:
                    safe_name = raw_name.replace('/', '_')
                    name = f'group{group_idx}_{safe_name}'
                param_names.append(name)

                x_tensor = None
                d_tensor = None
                grad_tensor = None
                true_grad_tensor = None
                x_shape = None
                d_shape = None
                grad_shape = None
                abs_err = None
                rel_err = None
                cosine_sim = None
                norm_ratio = None
                sign_agreement = None
                sign_agreement_n = None
                vector_size = None
                min_ratio = None
                max_ratio = None
                min_abs_ratio = None
                max_abs_ratio = None

                # Recompute x_input/d_input with the same path used in prepare_grad_old.
                x_input = None
                d_input = None
                if analog_ctx.has_gradient():
                    try:
                        x_input, d_input = self.get_processed_x_d_input(analog_ctx)
                        x_tensor = x_input.detach().cpu()
                        d_tensor = d_input.detach().cpu()
                        x_shape = tuple(x_tensor.shape)
                        d_shape = tuple(d_tensor.shape)
                    except Exception as e:
                        print(f"Warning: Could not process x/d input for {name}: {e}")
                else:
                    print(f"Warning: No cached analog gradient signals for {name}, skip x/d reconstruction.")

                # Recompute grad with prepare_grad_old-style sampling logic.
                if x_input is not None and d_input is not None:
                    try:
                        state = self.analog_state[param]
                        grad_tile = state["grad_tile"]

                        grad_sum = None
                        for _sample_idx in range(self.n_grad_sample):
                            if self.split_signal:
                                grad_sample = self._get_grad_by_rank_update_split_signal(
                                    grad_tile,
                                    x_input,
                                    d_input,
                                    shuffle_columns=self.shuffle_grad_columns,
                                )
                            else:
                                grad_sample = self._get_grad_by_rank_update(
                                    grad_tile,
                                    x_input,
                                    d_input,
                                    shuffle_columns=self.shuffle_grad_columns,
                                )

                            if grad_sum is None:
                                grad_sum = grad_sample
                            else:
                                grad_sum.add_(grad_sample)

                        if grad_sum is not None:
                            grad = grad_sum.div_(self.n_grad_sample)
                            grad_tensor = grad.detach().cpu()
                            grad_shape = tuple(grad_tensor.shape)
                    except Exception as e:
                        print(f"Warning: Could not recompute grad for {name}: {e}")

                # Fallback to current param.grad if recomputation failed.
                if grad_tensor is None and hasattr(param, 'grad') and param.grad is not None:
                    try:
                        grad_tensor = param.grad.detach().cpu()
                        grad_shape = tuple(grad_tensor.shape)
                    except Exception as e:
                        print(f"Warning: Could not process fallback grad for {name}: {e}")

                # Compute true gradient and error metrics
                if x_tensor is not None and d_tensor is not None:
                    try:
                        true_grad = torch.matmul(
                            d_tensor.view(-1, d_tensor.size(-1)).transpose(0, 1),
                            x_tensor.view(-1, x_tensor.size(-1))
                        )
                        true_grad_tensor = true_grad

                        if grad_tensor is None:
                            continue

                        abs_err = (true_grad - grad_tensor).norm().item()
                        true_grad_norm_val = true_grad.norm().item()
                        grad_norm_val = grad_tensor.norm().item()
                        rel_err = abs_err / (true_grad_norm_val + 1e-8)
                        cosine_sim = cosine_similarity(true_grad.flatten(), grad_tensor.flatten(), dim=0).item()

                        norm_ratio = grad_norm_val / (true_grad_norm_val + 1e-8)
                        sign_agreement_n = (torch.sign(grad_tensor) == torch.sign(true_grad)).sum().item()
                        vector_size = grad_tensor.numel()
                        sign_agreement = (
                            (torch.sign(grad_tensor) == torch.sign(true_grad)).float().mean().item()
                        )
                        element_wise_ratio = grad_tensor / (true_grad + true_grad.sign() * 1e-8)
                        max_ratio = element_wise_ratio.max().item()
                        min_ratio = element_wise_ratio.min().item()
                        min_abs_ratio = element_wise_ratio.abs().min().item()
                        max_abs_ratio = element_wise_ratio.abs().max().item()

                        print(
                            f"[{name}] relative error: {rel_err * 100:.3f}%, "
                            f"cosine: {cosine_sim:.6f}, "
                            f"norm ratio: {norm_ratio:.6f}, "
                            f"sign agreement: {sign_agreement * 100:.2f}% "
                            f"({sign_agreement_n}/{vector_size}), "
                            f"ratio: [{min_ratio:.3f}, {max_ratio:.3f}], "
                            f"abs_ratio: [{min_abs_ratio:.6f}, {max_abs_ratio:.6f}]"
                        )
                    except Exception as e:
                        print(f"Warning: Could not compute error metrics for {name}: {e}")
                
                # Collect error statistics
                if abs_err is not None:
                    grad_norm = grad_tensor.norm().item() if grad_tensor is not None else None
                    true_grad_norm = true_grad_tensor.norm().item() if true_grad_tensor is not None else None
                    error_stats.append({
                        'name': name,
                        'abs_err': abs_err,
                        'rel_err': rel_err,
                        'cosine_sim': cosine_sim,
                        'grad_norm': grad_norm,
                        'true_grad_norm': true_grad_norm,
                        'norm_ratio': norm_ratio,
                        'sign_agreement': sign_agreement,
                        'sign_agreement_n': sign_agreement_n,
                        'vector_size': vector_size,
                        'min_ratio': min_ratio,
                        'max_ratio': max_ratio,
                        'min_abs_ratio': min_abs_ratio,
                        'max_abs_ratio': max_abs_ratio,
                    })
                
                # Save in one file if any exists
                if x_tensor is not None or d_tensor is not None or grad_tensor is not None:
                    fname = f'{name}_signal.pt'
                    torch.save({
                        'x_input': x_tensor, 
                        'd_input': d_tensor, 
                        'grad': grad_tensor,
                        'true_grad': true_grad_tensor,
                        'abs_err': abs_err,
                        'rel_err': rel_err,
                        'cosine_sim': cosine_sim,
                        'norm_ratio': norm_ratio,
                        'sign_agreement': sign_agreement,
                        'sign_agreement_n': sign_agreement_n,
                        'vector_size': vector_size,
                        'min_ratio': min_ratio,
                        'max_ratio': max_ratio,
                        'min_abs_ratio': min_abs_ratio,
                        'max_abs_ratio': max_abs_ratio,
                    }, os.path.join(signal_dir, fname))
                    info = f'{fname}:'
                    if x_tensor is not None:
                        info += f' x_input_shape={x_shape}'
                    if d_tensor is not None:
                        info += f' d_input_shape={d_shape}'
                    if grad_tensor is not None:
                        info += f' grad_shape={grad_shape}'
                    if abs_err is not None:
                        info += (
                            f' abs_err={abs_err:.6e} rel_err={rel_err*100:.3f}% cosine={cosine_sim:.6f}'
                            f' norm_ratio={norm_ratio:.6f}'
                            f' sign_agreement={sign_agreement*100:.2f}% ({sign_agreement_n}/{vector_size})'
                            f' ratio=[{min_ratio:.3f}, {max_ratio:.3f}]'
                            f' abs_ratio=[{min_abs_ratio:.6f}, {max_abs_ratio:.6f}]'
                        )
                    tensor_info.append(info)
        
        # Write tensor info to list.log
        listlog_path = os.path.join(signal_dir, 'list.log')
        with open(listlog_path, 'w') as f:
            f.write("# Each .pt file contains a dict: {'x_input': tensor or None, 'd_input': tensor or None, 'grad': tensor or None, 'true_grad': tensor or None, 'abs_err': float or None, 'rel_err': float or None, 'cosine_sim': float or None}\n")
            f.write("# Format: filename: x_input_shape=(...), d_input_shape=(...), grad_shape=(...), abs_err=..., rel_err=...%, cosine=...\n")
            for line in tensor_info:
                print(line)
                f.write(line + '\n')
        
        # Write error statistics to error_stats.log
        if error_stats:
            error_stats_path = os.path.join(signal_dir, 'error_stats.log')
            with open(error_stats_path, 'w') as f:
                f.write("# Error statistics for each layer\n")
                f.write("# Format: name, abs_err, rel_err, cosine_sim, grad_norm, true_grad_norm, norm_ratio, sign_agreement, ratio_range, abs_ratio_range\n")
                for stats in error_stats:
                    line = (
                        f"{stats['name']}: "
                        f"abs_err={stats['abs_err']:.6e}, "
                        f"rel_err={stats['rel_err']*100:.3f}%, "
                        f"cosine={stats['cosine_sim']:.6f}, "
                        f"grad_norm={stats['grad_norm']:.6e}, "
                        f"true_grad_norm={stats['true_grad_norm']:.6e}, "
                        f"norm_ratio={stats['norm_ratio']:.6f}, "
                        f"sign_agreement={stats['sign_agreement']*100:.2f}% "
                        f"({stats['sign_agreement_n']}/{stats['vector_size']}), "
                        f"ratio=[{stats['min_ratio']:.3f}, {stats['max_ratio']:.3f}], "
                        f"abs_ratio=[{stats['min_abs_ratio']:.6f}, {stats['max_abs_ratio']:.6f}]"
                    )
                    print(line)
                    f.write(line + '\n')
            print(f"Error statistics written to {error_stats_path}")
        
        # Write parameter names to param_names.txt
        param_names_path = os.path.join(signal_dir, 'param_names.txt')
        with open(param_names_path, 'w') as f:
            for name in param_names:
                f.write(name + '\n')
        print(f"All analog x_input, d_input, and grad saved to {signal_dir}")
        print(f"Tensor info written to {listlog_path}")
        print(f"Parameter names written to {param_names_path}")


class AnalogGradMpMwOptimizerMixin(AnalogGradMpOptimizerMixin):
    """Mixed-weight variant of the analog mixed-precision optimizer.

    This mixin keeps an explicit real-weight matrix W copied from the
    analog tile and a perturbation matrix V. The effective (virtual)
    weight used by the framework is U = W + gamma * V. The analog tile's
    weights are set to U (via `set_weights`) so forward/backward use U.

    The method `compute_mixed_weight(gamma, V, W)` computes U and
    `update_mixed_weight_for_param` applies it to a specific layer.
    """

    def __init__(self, params, *args: Any, **kwargs: Any) -> None:
        # No gamma or V provided at init. W is copied but mixed weight U is
        # applied only when compute/update is explicitly called.
        super().__init__(params, *args, **kwargs)
        
        self.mode = "real-weight"  # other mode is "mixed-weight"

        # initialize mixed-weight buffers (only copy W)
        # For each analog param, copy the real analog weight W and create V if missing,
        # then set the tile weights to U = W + gamma * V so forward/backward use U.
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                analog_tile = analog_ctx.analog_tile

                # read real weight W from the analog tile
                main_weights = analog_tile.get_weights()[0]
                # main_analog_tile = AnalogMatrix(
                #     main_weights, rpu_config=analog_tile.rpu_config,
                # )
                # state["main_analog_tile"] = main_analog_tile

                # all state is stored on CPU
                state = self.analog_state[param]
                state["main_weights"] = main_weights.to(analog_tile.device)
                
        self.loss_gain = None
            
    def __del__(self):
        """Destructor: try to restore real weights stored in main_analog_tile back to tiles
        and free resources stored in analog_state. Silent on error to avoid exceptions
        during interpreter shutdown.
        """
        self.restore_real_weight_for_param()

    @no_grad()
    def normalize_grad(self):
        """No-op: gradient normalization is handled in prepare_grad."""
        
        accumulated_norm_square = 0.0
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                accumulated_norm_square += param.grad.norm().item() ** 2
        accumulated_norm_square = accumulated_norm_square ** 0.5
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                param.grad.div_(accumulated_norm_square + 1e-16)
                
        self.grad_norm = accumulated_norm_square

    @no_grad()
    def sign_grad(self):
        """But sign can't be implemented by analog hardware."""
        
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                param.grad.sign_()
                
    @no_grad()
    def construct_virtual_mixed_weight(self, gamma: float):
        """Compute virtual mixed-weight U = W + gamma * V.

        Args:
            gamma: scalar mixing coefficient.
            G: Gradient tensor stored in param.grad.

        Returns:
            U = W - gamma * G
        """
        
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                analog_tile = analog_ctx.analog_tile

                state = self.analog_state[param]
                
                # cache the original weights
                if self.mode == "real-weight":
                    # it is a copy on CPU
                    main_weights = analog_tile.get_weights()[0]
                    state["main_weights"] = main_weights
                elif self.mode == "mixed-weight":
                    main_weights = state.get("main_weights")
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                main_weights = main_weights.to(analog_tile.device)

                # Generally, RPU devices have max/min bounds 
                #   (except for FloatingPointRPUConfig) 
                # In the mixed-weight case, we regard the tile as virtual,
                #   so we set the bounds to inf/-inf
                # The original max/min bounds are stored in state
                hidden_params = analog_tile.get_hidden_parameters()
                if self.mode != "mixed-weight" \
                    and "max_bound" in hidden_params and "min_bound" in hidden_params:
                    state["max_bound"] = hidden_params["max_bound"]
                    state["min_bound"] = hidden_params["min_bound"]

                    # hidden_params['max_bound'] = torch.full_like(main_weights, float('inf'), device='cpu')
                    # hidden_params['min_bound'] = torch.full_like(main_weights, float('-inf'), device='cpu')
                    
                    # Determine the max/min bounds
                    # Note: They the bounds of RPU devices (conductance)
                    #       not the weights, which may have been scaled.
                    # max_scale = main_weights.abs().max() + gamma * param.grad.abs().max() + 1e-5
                    # max_scale = max_scale.item()
                    # hidden_params['max_bound'] = max_scale * torch.ones_like(hidden_params['max_bound'])
                    # hidden_params['min_bound'] = -max_scale * torch.ones_like(hidden_params['min_bound'])
                    increase_scale = 1+abs(gamma)
                    hidden_params['max_bound'] = increase_scale * torch.ones_like(hidden_params['max_bound'])
                    hidden_params['min_bound'] = -increase_scale * torch.ones_like(hidden_params['min_bound'])
                    analog_tile.set_hidden_parameters(hidden_params)
                    
                if gamma == 0.0 or param.grad is None:
                    mixed_weight = main_weights
                else:
                    mixed_weight = main_weights - gamma * param.grad
                
                AnalogGradMpMwOptimizerMixin.set_weights_with_scaling(
                    analog_tile, mixed_weight)
                
        if self.mode != "mixed-weight":
            self.mode = "mixed-weight"
    
    @no_grad()
    def restore_real_weight_for_param(self) -> None:
        """Restore the real weight W back to the analog tile for a given param."""
        
        if self.mode == "real-weight":
            return
        
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                analog_tile = analog_ctx.analog_tile
                
                state = self.analog_state[param]
                main_weights = state.get("main_weights")
                # no need to load on GPU
                # main_weights = main_weights.to(analog_tile.device)
                
                # recover the weight bounds
                hidden_params = analog_tile.get_hidden_parameters()
                if "max_bound" in hidden_params and "min_bound" in hidden_params:
                    hidden_params['max_bound'] = state["max_bound"]
                    hidden_params['min_bound'] = state["min_bound"]
                    analog_tile.set_hidden_parameters(hidden_params)
                    
                AnalogGradMpMwOptimizerMixin.set_weights_with_scaling(
                    analog_tile, main_weights)
                
        self.mode = "real-weight"
        
    @staticmethod
    def set_weights_with_scaling(analog_tile, weight: torch.Tensor) -> None:
        """Set the weight to the analog tile, handling weight scaling if needed."""
        
        weight_to_set = weight
        weight_scaling = analog_tile.get_scales()
        if weight_scaling is not None:
            ws = weight_scaling.view(-1, 1).to(weight.device).type_as(weight)
            mask = ws != 0
            ws_safe = ws.clone()
            ws_safe[~mask] = 1  # avoid division by zero; rows with zero scale remain unchanged
            weight_to_set = weight / ws_safe
        analog_tile.set_weights(weight_to_set)
        
    @no_grad()
    def prepare_grad(self) -> None:
        assert self.loss_gain is not None, "loss_gain must be set before calling prepare_grad"
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                param.grad = self.loss_gain * param.grad
        
        # Don't call the parent class prepare_grad, 
        #   which is done in the main training loop
        # return super().prepare_grad()
    
    def debug_update_bound(self, factor: float = 1.2):
        """Debug function to update the weight bounds by a factor."""
        
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                analog_tile = analog_ctx.analog_tile
                
                hidden_params = analog_tile.get_hidden_parameters()
                if "max_bound" in hidden_params and "min_bound" in hidden_params:
                    hidden_params['max_bound'] = factor * torch.ones_like(hidden_params['max_bound'])
                    hidden_params['min_bound'] = -factor * torch.ones_like(hidden_params['min_bound'])
                    analog_tile.set_hidden_parameters(hidden_params)


class RandomDirectionMpOptimizerMixin(AnalogMpOptimizerMixin):
    @no_grad()
    def prepare_grad(self) -> None:
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                
                # Call `update` in the tile.
                if not analog_ctx.has_gradient():
                    # Forward never used.
                    raise ValueError("what??")
                    return

                analog_tile = analog_ctx.analog_tile
                in_features = analog_tile.in_size
                out_features = analog_tile.out_size
                
                random_direction = torch.randn(out_features, in_features).to(
                    analog_tile.device)
                if param.size() != random_direction.size():
                    param.data = torch.zeros_like(random_direction)
                param.grad = random_direction
                
                
class ZerothOrderOptimizerMixin(AnalogGradMpMwOptimizerMixin):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
    
    @no_grad()
    def prepare_ZO_grad(self, ZO_closure, **kwargs):
        # Initialize ZO_grad_buffer
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                    
                analog_ctx = param
                analog_tile = analog_ctx.analog_tile
                in_features = analog_tile.in_size
                out_features = analog_tile.out_size
                
                if not hasattr(param, 'ZO_grad_buffer') or param.ZO_grad_buffer is None:
                    param.ZO_grad_buffer = torch.zeros(
                        out_features, in_features).to(analog_tile.device)
                    
        # Accumulate ZO gradients
        n_repeat = 100
        for _ in range(n_repeat):
            self._prepare_ZO_grad_(ZO_closure)
            
        # Average the ZO gradients
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                    
                param.ZO_grad_buffer.div_(n_repeat)
                param.grad = param.ZO_grad_buffer
                del param.ZO_grad_buffer
            
        self.restore_real_weight_for_param()
    
    @no_grad()
    def _prepare_ZO_grad_(self, ZO_closure) -> None:
        with torch.no_grad():
            lr = self.param_groups[0]['lr']
            
            self.prepare_perturbation()
            # dynamic_ZO_stepsize = lr
            # dynamic_ZO_stepsize = 0.01 # fastest
            dynamic_ZO_stepsize = lr
            self.construct_virtual_mixed_weight(-dynamic_ZO_stepsize)
            _, loss_p = ZO_closure()
            self.construct_virtual_mixed_weight(dynamic_ZO_stepsize)
            _, loss_n = ZO_closure()
            self.loss_gain = (loss_p - loss_n) / (2 * dynamic_ZO_stepsize)
            
            for group in self.param_groups:
                # Use analog_tile object.
                for param in group["params"]:
                    if not isinstance(param, AnalogContext):
                        continue
                    
                    param.ZO_grad_buffer.add_(param.grad * self.loss_gain)
    
    @no_grad()
    def prepare_perturbation(self) -> None:
        # AnalogMpOptimizerMixin.prepare_grad(self)
        # return
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                
                # Call `update` in the tile.
                if not analog_ctx.has_gradient():
                    # Forward never used.
                    raise ValueError("what??")
                    return

                analog_tile = analog_ctx.analog_tile
                in_features = analog_tile.in_size
                out_features = analog_tile.out_size
                
                random_direction = torch.randn(out_features, in_features).to(
                    analog_tile.device)
                if param.size() != random_direction.size():
                    param.data = torch.zeros_like(random_direction)
                param.grad = random_direction

