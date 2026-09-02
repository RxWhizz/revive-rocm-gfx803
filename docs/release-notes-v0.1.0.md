# revive-rocm-gfx803 v0.1.0

Initial public-ready release for reviving AMD Polaris / gfx803 GPUs for small
FP32 AI workloads with ROCm 5.7.1, rocBLAS gfx803, PyTorch built from source,
and MACE.

## Highlights

- Docker-based ROCm 5.7.1 runtime.
- rocBLAS rebuilt with `gfx803` Tensile support.
- PyTorch 2.2.2 built from source with `PYTORCH_ROCM_ARCH=gfx803`.
- Conservative FP32 workflow for old Polaris cards.
- One-process-per-GPU isolation with `HIP_VISIBLE_DEVICES`.
- MACE import, inference, and tiny training validation.
- Dual 30 minute stress test completed with 0 NaN on two RX 570 cards.

## Validation Evidence

The release includes:

- `revive-rocm-gfx803-validation-v0.1.0.zip`
- `SHA256SUMS-v0.1.0.txt`

The archive contains the original validation logs, benchmark traces, checkpoint
artifacts, and status report captured from the validated host.

## Supported Scope

Tested:

- 2x Radeon RX 570, gfx803, device id `1002:67df`
- Ubuntu 24.04 host, kernel 6.17
- ROCm 5.7.1 user-space in Docker
- PyTorch 2.2.2 source build
- MACE 0.3.6

Expected but not yet validated by this repo:

- RX 470, RX 480, RX 580, RX 590
- Similar Polaris / gfx803 boards where KFD exposes the GPU correctly

## Install

```bash
git clone https://github.com/RxWhizz/revive-rocm-gfx803.git
cd revive-rocm-gfx803
make audit
make build
make test
make mace-test
```

Read the README before building. The first build can take hours and needs a
large Docker data area.
