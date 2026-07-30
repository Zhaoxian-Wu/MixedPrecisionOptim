import warnings
from typing import Any, Callable, Optional

import torch
from torch import cat
from torch.autograd import no_grad

from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs.compounds import UnitCell
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
        
        # initalize the digital buffer
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                
                if hasattr(param, "enable_buffer"):
                    param.enable_buffer()
                else:
                    mes = "AnalogContext does not have `enable_buffer` method."
                    warnings.warn(mes, UserWarning)

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
        """Compute gradient via digital MAC operations.

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
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param
                
                # Call `update` in the tile.
                if not analog_ctx.has_gradient() and analog_ctx.grad is None:
                    raise ValueError("No gradient found in AnalogContext. backward() might not have been called.")

                x_input, d_input = self.get_processed_x_d_input(analog_ctx)
                grad = self.compute_gradient(x_input, d_input)
                
                # TODO: use param as transfer buffer
                
                if param.size() != grad.size():
                    # for the latest aihwkit, it param has already been initialized
                    assert not hasattr(param, "enable_buffer")
                    # AnalogContext.data starts as a scalar (aihwkit context.py:
                    # ones(())). The digital transfer buffer is lazily initialized
                    # to the gradient/weight shape on the first step. This replaces
                    # a temporary debug `raise` and substitutes for
                    # AnalogContext.enable_buffer(), which the current aihwkit build
                    # does not provide (see __init__ warning).
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