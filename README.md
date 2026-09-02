# revive-rocm-gfx803

Bring old AMD Polaris / gfx803 GPUs back to useful life for small AI training
and inference with PyTorch, ROCm, Docker, and MACE.

[![ROCm](https://img.shields.io/badge/ROCm-5.7.1-red)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2%20source-orange)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-gfx803%20Polaris-blue)](docs/compatibility.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This project is for users who own cards such as RX 470, RX 480, RX 570,
RX 580, RX 590, MI6, or MI8 and want a reproducible path to run FP32 PyTorch
workloads after modern ROCm support stopped being straightforward for gfx803.

The approach is intentionally conservative:

- Linux host provides `amdgpu`, KFD, `/dev/kfd`, `/dev/dri`, and Docker.
- Docker image provides ROCm 5.7.1 user-space.
- rocBLAS is rebuilt with Tensile support for `gfx803`.
- PyTorch is built from source with `PYTORCH_ROCM_ARCH=gfx803`.
- AI workloads run in FP32, one process per GPU.
- No FP16, BF16, Triton, `torch.compile`, DDP, or VRAM pooling is promised.

## Current Status

Validated on one machine with two AMD RX 570 cards:

| Check | Result |
|---|---|
| Host detects 2 Polaris / gfx803 GPUs | PASS |
| ROCm 5.7.1 container sees both GPUs | PASS |
| rocBLAS has gfx803 kernels | PASS |
| PyTorch imports and sees HIP | PASS |
| FP32 matmul matches CPU sample | PASS |
| Backward pass has finite gradients | PASS |
| Adam training lowers loss | PASS |
| Two isolated GPU processes run at once | PASS |
| MACE imports | PASS |
| MACE inference on GPU | PASS |
| MACE tiny training on GPU | PASS |
| Dual 30 minute stress test | PASS, 0 NaN |

Evidence summaries are kept in `results/`. Full logs, long benchmark traces,
and checkpoints are attached to the `v0.1.0` release instead of being stored in
the Git tree.

## Quickstart

Test the host first. The audit is read-only and writes `logs/host_audit.txt`.

```bash
git clone https://github.com/RxWhizz/revive-rocm-gfx803.git
cd revive-rocm-gfx803
make audit
```

If the gate passes, build the runtime image:

```bash
make build
```

The build can take hours because it rebuilds rocBLAS and PyTorch from source.
See [build time and disk](docs/build-time-and-disk.md) before starting on a
small SSD.

Run the core validation:

```bash
make detect
make test
make test-gpu0
make test-gpu1
make test-dual
make mace-test
```

Optional stress and benchmark runs:

```bash
make benchmark
make matmul-ab
make sgemm-ab
make gemm-shapes
```

## Requirements

Known-good host:

- Ubuntu 24.04
- Linux kernel 6.17
- Two RX 570 GPUs, device id `1002:67df`
- 62 GiB RAM
- Docker with BuildKit
- User in the `render`, `video`, and `docker` groups

Minimum practical target:

- Linux host with working `amdgpu` and KFD
- One or more gfx803 GPUs
- 4 GiB VRAM per GPU for small tests; 8 GiB is more comfortable
- 16 GiB RAM minimum; 32 GiB or more recommended for building
- Large free disk area for Docker layers and source builds

See [compatibility](docs/compatibility.md) for what is tested, expected, and
unknown.

## Architecture

```text
Linux host
  amdgpu + KFD + /dev/kfd + /dev/dri + Docker
    |
    v
Docker image: rocm/dev-ubuntu-22.04:5.7.1
  rocBLAS rebuilt for gfx803
  PyTorch 2.2.2 built from source for gfx803
  micromamba Python 3.10 runtime
    |
    v
FP32 workloads
  PyTorch tests
  dual GPU isolation
  MACE inference and tiny training
```

## Why This Exists

Official wheels for modern PyTorch on ROCm do not carry the right kernels for
many Polaris / gfx803 cards. A typical failure is:

```text
RuntimeError: HIP error: no kernel image is available for execution on the device
```

This repo avoids that path by rebuilding the pieces that need gfx803 support.
It also disables hipBLASLt preference because hipBLASLt is not the right route
for Polaris:

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_ROCM_ARCH=gfx803
```

AMD still documents `gfx803` as an LLVM target for GPUs such as MI6 and MI8,
but current mainstream ROCm support is focused elsewhere. Treat this repository
as an unsupported community revival path, not an official AMD support channel.

Useful references:

- AMD ROCm compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- AMD GPU specifications: https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html

## Make Targets

| Target | Purpose |
|---|---|
| `make audit` | Read-only host audit and gate check |
| `make build` | Build final runtime image |
| `make build-base` | Build only the base stage |
| `make build-rocblas` | Build through rocBLAS gfx803 stage |
| `make build-pytorch` | Build through PyTorch source stage |
| `make shell` | Enter the runtime container |
| `make detect` | Print PyTorch/HIP/GPU information |
| `make test` | Run core FP32 PyTorch tests |
| `make test-gpu0` | Validate isolated GPU 0 process |
| `make test-gpu1` | Validate isolated GPU 1 process |
| `make test-dual` | Run two isolated jobs at once |
| `make benchmark` | Run 30 minute dual stress test |
| `make matmul-ab` | Benchmark PyTorch/rocBLAS matmul |
| `make sgemm-ab` | Benchmark standalone HIP SGEMM variants |
| `make gemm-shapes` | Capture GEMM shapes from MACE |
| `make mace-test` | Run MACE import, inference, and tiny training |
| `make report` | Print collected logs and summaries |

## Expected Results

On the validated RX 570 host:

- `torch.cuda.device_count()` reports 2 GPUs.
- `HIP_VISIBLE_DEVICES=0` and `HIP_VISIBLE_DEVICES=1` each expose exactly one
  GPU to one process.
- FP32 `torch.matmul` on 4096 square matrices reaches about 1.25 TFLOPS per
  RX 570 and matches a CPU sample.
- Dual 30 minute stress test completed with 0 NaN on both GPUs.
- MACE 0.3.6 imports, runs inference, and completes a tiny 3 epoch training
  run on one GPU.

Your numbers may differ because Polaris cards vary in VRAM, clocks, cooling,
PCIe link, power limit, and kernel/driver behavior.

## Troubleshooting

Start here:

- [Troubleshooting guide](docs/troubleshooting.md)
- [Compatibility matrix](docs/compatibility.md)
- [Build time and disk planning](docs/build-time-and-disk.md)
- [GEMM optimization notes](docs/gemm_gfx803_optimization.md)

When opening an issue, include:

```bash
make audit
make detect
make test
```

Attach `logs/host_audit.txt`, the failing command, and the last 80 lines of the
relevant build or runtime log.

## What This Does Not Do

- It does not install PyTorch, MACE, or ROCm Python packages on the host.
- It does not make AMD Polaris officially supported by AMD.
- It does not combine VRAM across cards.
- It does not promise modern ROCm features such as FP16/BF16 acceleration,
  Triton, DDP, or `torch.compile`.
- It does not target Windows.

## License

MIT. See [LICENSE](LICENSE).
