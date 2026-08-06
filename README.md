# MixedPrecisionOptim

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

Currently the following mixed-precision optimizers are tracked for `A01-train-analog.py`:

- `MpAdamW`
- `MpMuon`

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
optimizer = MpAdamW(params, lr=lr, betas=(beta1, beta2))

# required by aihwkit
if hasattr(optimizer, "regroup_param_groups"):
    optimizer.regroup_param_groups(model)
```
To use Muon on the analog mixed-precision path, replace the constructor:

```python
from MixedPrecisionOptim.Muon import MpMuon
optimizer = MpMuon(params)
```

## Notes

- Call `optimizer.regroup_param_groups(model)` after construction when the
  optimizer exposes that method. This isolates analog parameters into dedicated
  groups and synchronizes tile learning rates.
- `MpAdamW` currently does **not** provide the weight decay mechanism on the
  analog mixed-precision path. The `weight_decay` interface is kept for
  compatibility, and a proper implementation is left for future work.
