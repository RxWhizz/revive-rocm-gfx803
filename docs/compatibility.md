# Compatibility

This project targets AMD GCN4 Polaris / gfx803 GPUs. It is a community revival
path for unsupported or difficult ROCm setups, not an official AMD support
statement.

## Tested

| Hardware | Device id | VRAM | Host | Result |
|---|---:|---:|---|---|
| Radeon RX 570 | `1002:67df` | 4 GiB | Ubuntu 24.04, kernel 6.17 | PASS |
| Radeon RX 570 | `1002:67df` | 8 GiB | Ubuntu 24.04, kernel 6.17 | PASS |

Validated workload:

- ROCm 5.7.1 user-space in Docker.
- rocBLAS rebuilt with Tensile support for `gfx803`.
- PyTorch 2.2.2 built from source with `PYTORCH_ROCM_ARCH=gfx803`.
- FP32 matmul, backward, Adam, dual process isolation, MACE import, MACE
  inference, MACE tiny training, and dual 30 minute stress test.

## Expected To Be Similar

These cards are commonly Polaris / gfx803, but this repo has not validated
every board revision:

| Family | Examples | Notes |
|---|---|---|
| Radeon RX 400 | RX 470, RX 480 | Likely same revival path; VRAM may be the limiting factor |
| Radeon RX 500 | RX 570, RX 580, RX 590 | Main target family |
| Radeon Pro Polaris | WX 5100, WX 7100 | May work if KFD exposes the device correctly |
| Instinct GCN gfx803 | MI6, MI8 | AMD documents gfx803 targets; exact container behavior still needs user reports |

## Not Targeted

| Hardware | Reason |
|---|---|
| RDNA / Navi | Different gfx targets and different ROCm behavior |
| Vega / gfx900, gfx906, gfx908 | Easier path exists; this repo may be adapted later |
| APUs | KFD/HIP behavior is highly system dependent |
| Windows | This workflow depends on Linux KFD and Docker device passthrough |

## Host Checklist

`make audit` must pass these gates:

- `amdgpu` loaded.
- `/dev/kfd` exists.
- At least one `/dev/dri/renderD*` node exists.
- Docker is installed.
- Current user belongs to `render`, `video`, and `docker`.

For multi-GPU use, each physical GPU must have a render node and must be
selectable through `HIP_VISIBLE_DEVICES`.

## Reporting A New Card

Open an issue with:

```bash
lspci -nnk | grep -A3 -iE "VGA|3D controller|Display"
make audit
make detect
make test
```

Include card model, VRAM size, distro, kernel, Docker version, and whether the
runtime needed `HSA_OVERRIDE_GFX_VERSION=8.0.3`.
