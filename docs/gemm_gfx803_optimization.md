# FP32 GEMM On RX 570 / gfx803

This note explains how to check whether the revived ROCm stack is using the
GPU correctly and how to collect the data needed for future GEMM tuning.

## Current Baseline

On the validated RX 570 host:

- PyTorch sees two gfx803 GPUs.
- `HIP_VISIBLE_DEVICES=0` and `HIP_VISIBLE_DEVICES=1` isolate one process per
  card.
- `torch.matmul` through rocBLAS reaches about 1.25 TFLOPS per RX 570 on
  `4096x4096x4096` FP32 GEMM.
- The result matches a CPU sample and produces finite values.

That is enough to prove the stack is usable, but it is not close to theoretical
peak. A rough RX 570 FP32 peak estimate is:

```text
32 CU * 64 lanes * 2 FMA * 1.244 GHz = about 5.09 TFLOPS
```

So 1.25 TFLOPS is about 25 percent of peak. For large GEMMs this suggests that
kernel selection, Tensile logic, clocks, or occupancy may matter more than raw
memory bandwidth alone.

## Check Both GPUs

A single PyTorch process does not combine two RX 570 cards. This project uses
one process per GPU:

```bash
make detect
make test-gpu0
make test-gpu1
make test-dual
make matmul-ab
```

`make matmul-ab` launches two simultaneous processes and writes:

- `results/matmul_ab_gpu0.json`
- `results/matmul_ab_gpu1.json`
- `results/matmul_ab_summary.txt`

If both JSON files exist and the summary names both GPUs, the benchmark used
both cards through separate processes.

## Benchmark Layers

Use two layers:

```bash
make matmul-ab
make sgemm-ab
```

Interpretation:

- If PyTorch/rocBLAS clearly beats the standalone kernels, rocBLAS is working
  and the remaining question is tuning for your shapes.
- If the LDS kernel beats the conservative kernel, memory reuse is helping.
- If the `vec4` kernel only helps when dimensions are multiples of four, vector
  loads are useful only for aligned shapes.
- If everything is slow on large matrices, inspect clocks, temperature, power
  limit, and PCIe link with `rocm-smi` and `lspci`.

## Standalone Kernels

File:

```text
benchmarks/sgemm_gfx803_hip.cpp
```

Variants:

| Variant | Purpose |
|---|---|
| `conservative` | One thread computes one `C` element; correctness baseline |
| `lds` | Uses 16x16 LDS/shared-memory tiles |
| `vec4` | Computes four contiguous columns with `float4` loads |

Manual run inside the container:

```bash
bash scripts/build_sgemm_kernels.sh
/workspace/results/sgemm_gfx803_hip --shape 4096 --repeats 20 --warmup 5 --csv /workspace/results/sgemm_ab.csv
```

## Capture Real GEMM Shapes

Do not tune only for square 4096 GEMMs if your real workload uses narrow or
batched shapes. Capture the actual rocBLAS calls first:

```bash
make gemm-shapes
```

Default workload:

```bash
ROCBLAS_LAYER=2 python mace/test_mace_inference.py
```

Outputs:

- `results/gemm_shapes_gpu0.log`
- `results/gemm_shapes_gpu0.csv`
- `results/gemm_shapes_gpu0_summary.txt`

For another workload:

```bash
GPU=0 bash scripts/capture_gemm_shapes.sh "python mace/test_mace_training.py"
```

## Future Tensile Tuning

Template:

```text
configs/tensile_gfx803_sgemm_template.yaml
```

Recommended flow:

1. Capture real shapes with `make gemm-shapes`.
2. Copy dominant shapes into `BenchmarkFinalParameters`.
3. Run Tensile on real gfx803 hardware.
4. Install the generated logic into rocBLAS or build a separate library.
5. Rebuild the runtime and rerun `make matmul-ab`.

The goal is not to optimize a synthetic square GEMM if MACE or ALIGNN uses a
different distribution. Optimize the shapes your workload actually calls.
