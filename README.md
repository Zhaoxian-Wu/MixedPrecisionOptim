# MixedPrecisionOptim

`test-v3` is the unified optimizer code snapshot with checkpoint format v3.

`MixedPrecisionOptim` is an enhanced mixed-precision training package built on
top of [IBM/aihwkit](https://github.com/IBM/aihwkit). In particular, it extends
the mixed-precision training flow provided by AIHWKIT components such as
`DigitalRankUpdateRPUConfig` and `MixedPrecisionCompound`, with additional
support for modern optimizers used in current training workloads.

This project is built upon AIHWKIT, and its optimizer interface is designed to
be completely compatible with the standard PyTorch optimizer interface. In
practice, these optimizers can be constructed and used like regular PyTorch
optimizers, while adding analog-aware mixed-precision update behavior for
AIHWKIT-based models.

The shared base is `MpOptimizerMixin`. Public adapters are `MpAdamW`,
`MpSGD`, `MpMuon`, `MpMoonlight`, `MpLion`, `MpShampoo`, and `MpRMSprop`.
Each lives in its algorithm's module. The class factory `as_logical_mp` is
retained and generates names with the `Mp` prefix. Former class and module
names are removed; no compatibility aliases are exported.

## Example Usage


```python
import torch

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import SoftBoundsReferenceDevice

from MixedPrecisionOptim.AdamW import MpAdamW

# ==========================================
# RPU configuration set as needed
# ==========================================
rpu_config = SingleRPUConfig(
    device=SoftBoundsReferenceDevice(
        dw_min=0.001,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
    )
)
rpu_config.forward.inp_res = 1 / (2**8 - 2)
rpu_config.forward.out_res = 1 / (2**8 - 2)
rpu_config.forward.inp_noise = 0.0
rpu_config.forward.out_noise = 0.0
rpu_config.backward.inp_res = 1 / (2**8 - 2)
rpu_config.backward.out_res = 1 / (2**8 - 2)

rpu_config.backward.inp_noise = 0.0
rpu_config.backward.out_noise = 0.0
# ==========================================


# create a simple model with a single linear layer
model = torch.nn.Sequential(
  torch.nn.Linear(128, 64),
)

model = convert_to_analog(model, rpu_config)

lr = 1e-3
beta1 = 0.9
beta2 = 0.999

params = model.parameters()
optimizer = MpAdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=0.0)

# MpOptimizerMixin automatically isolates analog parameter groups during construction.
```
To use Muon on the analog mixed-precision path, replace the constructor:

```python
from MixedPrecisionOptim.Muon import MpMuon
optimizer = MpMuon([
    dict(params=matrix_params, use_muon=True, lr=0.02, momentum=0.95),
    dict(params=auxiliary_params, use_muon=False, lr=3e-4,
         betas=(0.9, 0.95), eps=1e-10),
], weight_decay=0.0)
```

## Notes

- Construction automatically isolates analog parameters into dedicated groups,
  preserves their group settings, and synchronizes tile learning rates.
  `regroup_param_groups(model)` remains available for explicitly regrouping and
  recording parameter names, but normal initialization requires no extra call.
- Weight decay is temporarily disabled in all project MP wrappers, including
  `MpAdamW`, `MpSGD`, `MpMuon`, and the other MP adapters. Every parameter group
  must use `weight_decay=0.0`; nonzero decay raises `NotImplementedError` during
  construction, group addition, and checkpoint saving/loading. Training steps
  do not repeat this check; do not enable decay by editing parameter-group
  dictionaries or from an optimizer hook/closure.
  Native defaults are preserved, so explicitly pass zero when the wrapped
  optimizer defaults to nonzero decay (for example, PyTorch AdamW).

## MP optimizer mixin

`MpOptimizerMixin` owns mapped-weight gradients, the pending-update residual, quantized
array writes, checkpoint validation, and the centralized DDP update protocol.
It calls the wrapped PyTorch optimizer's `step()` on ordinary proxy parameters;
it contains no Adam, SGD, Muon, or other optimizer recurrence.

Wrap an optimizer **class**, then construct it with the usual PyTorch arguments:

```python
from MixedPrecisionOptim import as_logical_mp

MpAdam = as_logical_mp(torch.optim.Adam)
optimizer = MpAdam(model.parameters(), lr=1e-3, weight_decay=0.0)
```

The model must already use the project's mapped-weight parameterization: contexts
use native mapping scales, and gradients must target
`W = s * W_hat`. Native output alpha remains an outer factor.
This adapter requires the native mapped-weight gradient API used by `load_env.sh`.

All MP adapters run the digital rule directly on the pending-update
residual. Each ordinary proxy parameter shares the residual's storage. The
adapter stores no digital weight shadow or before-step weight snapshot and
does not read array weights to initialize optimizer state. Native mapping
warm-up is supported.

The wrapped rule's update increment must depend only on gradients and optimizer
state, not current weight values. Supported examples include AdamW at zero
decay and the project's Muon/Moonlight updates. Rules that use weight values
to compute an update are unsupported even if they expose no `weight_decay`
option; the adapter does not automatically detect such custom rules. Nonzero
decay is rejected for every analog and digital parameter group.

Every analog residual lives in `AnalogContext`'s BUFFER; proxy parameters and
`mp_residual` checkpoint metadata refer to that same storage. The
`disable_context_buffer` option has been removed. Linear and indexed Conv
layers accumulate gradients through AIHWKit's native backward API; retained
legacy x/d traces can also be reconstructed. Both use the existing DDP
protocol: call `prepare_for_ddp(model)` before creating DDP and
`ddp_synchronize_gradients(ddp_model)` before each step.

Tiles without a gradient this step keep their pending residual until a later
step with a gradient. After all writes, each tile receives one
`post_update_step()` call, including inactive tiles, matching the ordinary MP
mixin's per-step diffusion/decay policy. Frozen analog weights skip both writes
and post-update operations. Under DDP only rank 0 advances physical tile state.

Full optimizer-state checks run when MP state is first initialized or a
checkpoint is loaded. Normal steps do not scan residuals for NaN/Inf or compare
checkpoint scales. The current programming scale `s` is still obtained from
mapping metadata for the pulse quantum `abs(s) * dw_min` and conversion between
mapped-weight and physical increments; it does not read array weights. Static write
configuration is checked at construction, group registration, and restoration.

There is one MP implementation: `MpOptimizerMixin`. AdamW has one public adapter,
`MpAdamW`, implemented in `AdamW.py`. SGD and Muon expose the canonical
`MpSGD` and `MpMuon` adapters. Muon accepts ordinary parameter
lists or explicit `use_muon` parameter groups with per-group hyperparameters.
Training entry points and protocol files use the same `Mp*` names.
`TRAINING_ALG="AIHWKitMpSGD"` selects
AIHWKit's `AnalogSGD` with `MixedPrecisionCompound`, independently of this
library's `MpSGD` adapter. MVM wrapping is selected independently by
`SCALED_ANALOG_MVM`: formal ablation profiles set this flag, while manual runs
set it explicitly. For example, `MpAdamW` with the `original` profile keeps
ordinary AIHWKit MVMs; the same optimizer with the `s_out` profile uses mapped-weight
MVM wrappers. Optimizer names alone never enable those wrappers.
Muon retains the established per-group settings for each model configuration.
The project's additional log-scale gain and `s_ls` profile have been removed;
learnable output scaling uses AIHWKit's native `out_scaling_alpha`.
Subclasses inherit `MpOptimizerMixin` and set
`digital_optimizer_cls` rather than assigning `torch_step`.

Each algorithm has its own module containing its digital rule or native import
and its MP wrapper.

Custom digital rules use algorithm-specific constructors. Lion accepts
`lr`, `betas`, and `weight_decay`; its existing default betas remain
`(0.9, 0.999)`. Shampoo accepts `lr`, `shampoo_beta`, `eps`, and `weight_decay`;
`shampoo_beta=0.99` controls covariance accumulation. Unsupported options are
rejected at construction, group addition, and checkpoint loading. Group
metadata such as names and base learning rates is preserved.

`_DigitalRuleOptimizer` inherits PyTorch's `(params, defaults)` constructor
and supplies only shared update lifecycle and group-validation hooks.
Muon/Moonlight inherit PyTorch `Optimizer` directly. Their matrix options
are passed to native Muon; `betas`, `eps`, and AdamW options belong to the
auxiliary optimizer.

| Module | Digital implementation |
|---|---|
| `AdamW.py` | `torch.optim.AdamW` |
| `SGD.py` | `torch.optim.SGD` |
| `RMSprop.py` | `torch.optim.RMSprop`, including momentum/centering |
| `Muon.py` | `torch.optim.Muon` plus `torch.optim.AdamW` for auxiliaries |
| `Moonlight.py` | Native Muon with `adjust_lr_fn="match_rms_adamw"`, plus native AdamW |
| `Lion.py` | Project Lion rule |
| `Shampoo.py` | Project two-sided Shampoo rule |

`MpOptimizers.py` provides convenience exports, and `digital_optimizers.py`
contains digital-rule imports only. `MpOptimizerMixin` also owns parameter-group
regrouping and tile learning-rate synchronization. Direct analog X/D update
algorithms use AIHWKit's optimizers.

The installed PyTorch environment provides native Muon. Matrix parameters
with more than two dimensions are flattened for its 2D API; contiguous views
share storage, while non-contiguous weights require a temporary contiguous
matrix and copy-back. Native rules own Newton–Schulz and Adam recurrences.
PyTorch's Muon kernels can round differently from the former handwritten
BF16 equations, so historical trajectories are not promised bit-for-bit.
The native rule skips parameters with no gradient, including auxiliary groups.

SGD and RMSprop now use PyTorch constructor defaults and accepted arguments;
RMSprop does not accept Adam's `betas` or `fused`. Former RMSprop checkpoint
states missing native bookkeeping must be repaired offline before loading.

Standard PyTorch Adafactor always uses current weight RMS to scale its step.
The residual-only adapter cannot substitute BUFFER for that weight, so
`as_logical_mp(torch.optim.Adafactor)` raises `NotImplementedError`.

Save and restore the model and optimizer together with their `state_dict()`
interfaces. The loader accepts current v3 MP checkpoints without
historical format conversion. Scales remain exclusively on the model tiles;
`mp_state()` returns only BUFFER residual and transfer count.
MP metadata uses `mp_version=3`, `mp_residual`, and `mp_transfer_steps`.
Historical checkpoint repair requirements are recorded in the enclosing
project's `notes/2026-10-11-logical-mp-checkpoint-repair.md`.
Array writes are implemented by `MpOptimizerMixin._update_mapping_tile()`, which
submits mapped-weight increments through native mapping preprocessing and
`tile.update()`. Learning-rate schedulers and step hooks operate on the public
adapter. A supplied closure runs once on
the live analog model before gradient collection. Multi-evaluation optimizers
such as LBFGS, differentiable steps, optimizers caching their own parameter
lists outside `param_groups`, and group-topology changes during `step()` are
unsupported. For DDP, evaluate closures and synchronize gradients explicitly
before calling `step()`.
