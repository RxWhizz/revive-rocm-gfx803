# Status Report

Repository public-readiness update: 2026-09-01.
Validation evidence captured on the original RX 570 host: 2026-06-29.

## Summary

The validated stack works on the tested host:

- 2x AMD Radeon RX 570 Polaris / gfx803 detected.
- ROCm 5.7.1 user-space runs inside Docker.
- rocBLAS was rebuilt with gfx803 Tensile support.
- PyTorch 2.2.2 was built from source for gfx803.
- MACE 0.3.6 imports, runs inference, and completes tiny training.
- Dual 30 minute stress completed with 0 NaN on both GPUs.

This is an unsupported community revival path. It is not official AMD support.

## Tested Host

| Item | Value |
|---|---|
| GPUs | 2x AMD Ellesmere/Polaris, device id `1002:67df`, gfx803 |
| Cards | RX 570 4 GiB and RX 570 8 GiB |
| Driver | `amdgpu` loaded |
| Compute node | `/dev/kfd` present |
| Render nodes | `renderD128`, `renderD129` |
| OS / kernel | Ubuntu 24.04.4, kernel 6.17 |
| RAM / CPU | 62 GiB RAM, 88 threads |
| Docker | Gate passed after Docker/groups were configured |

## Validation Results

| Check | GPU 0 | GPU 1 | Result |
|---|---|---|---|
| PyTorch sees gfx803 | PASS | PASS | OK |
| FP32 matmul | PASS | PASS | OK |
| Backward gradients finite | PASS | Not run | OK |
| Adam loss decreases | PASS | PASS | OK |
| `HIP_VISIBLE_DEVICES` isolation | PASS | PASS | OK |
| Dual simultaneous jobs | PASS | PASS | OK |
| MACE import | PASS | PASS | OK |
| MACE inference | PASS | Not run | OK |
| MACE tiny training | PASS | Not run | OK |
| 30 minute dual stress | 286016 iterations, 0 NaN | 285401 iterations, 0 NaN | OK |

## Version Strategy

| Component | Choice |
|---|---|
| Base image | `rocm/dev-ubuntu-22.04:5.7.1` |
| ROCm user-space | 5.7.1 |
| Python | 3.10 |
| rocBLAS | `release/rocm-rel-5.7`, rebuilt for gfx803 |
| PyTorch | 2.2.2, source build |
| NumPy | 1.26.4 |
| ASE | 3.22.1 |
| MACE | 0.3.6 |

## Important Technical Notes

- `PYTORCH_ROCM_ARCH=gfx803` is required during the PyTorch build.
- `TORCH_BLAS_PREFER_HIPBLASLT=0` is required because Polaris should use the
  classic rocBLAS path.
- `apt-mark hold rocblas rocblas-dev` protects the rebuilt gfx803 rocBLAS
  packages while installing additional ROCm libraries.
- The validated host did not require `HSA_OVERRIDE_GFX_VERSION=8.0.3`, but the
  override remains documented as a diagnostic option.
- A single PyTorch process does not pool both GPUs. The project uses one
  process per GPU with `HIP_VISIBLE_DEVICES`.

## Evidence Location

Tracked in Git:

- `results/benchmark_gpu0_summary.json`
- `results/benchmark_gpu1_summary.json`
- `STATUS_REPORT.md`
- `docs/`

Stored as release assets:

- Full build logs
- Host audit log
- Long benchmark JSONL traces
- Stress-test checkpoints

## Public Readiness

Completed:

- README rewritten for public users.
- Troubleshooting guide added.
- Compatibility guide added.
- Build time and disk guide added.
- Large generated artifacts ignored going forward.
- Lightweight CI added for syntax and documentation presence.
- Issue template added for GPU compatibility reports.

Remaining manual GitHub settings:

- Make repository public.
- Add repository description and topics from `docs/publication-checklist.md`.
- Enable Issues if not already enabled.
