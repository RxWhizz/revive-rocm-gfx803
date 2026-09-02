# Build Time And Disk Planning

This repository rebuilds rocBLAS and PyTorch for `gfx803`. That is the point of
the project, but it means the first build is not quick.

## Why It Takes So Long

Official wheels are not enough for this target. The image builds:

1. A ROCm 5.7.1 base environment.
2. rocBLAS from source with Tensile logic for `gfx803`.
3. PyTorch 2.2.2 from source with `PYTORCH_ROCM_ARCH=gfx803`.
4. A runtime environment with ASE and MACE.

## Practical Expectations

| Resource | Recommendation |
|---|---|
| RAM | 32 GiB recommended, 16 GiB minimum with fewer jobs |
| CPU | More cores help; PyTorch build is CPU-heavy |
| Disk | Use a large Docker data-root; source layers can be many tens of GiB |
| Time | Expect hours for the first complete build |
| Network | Several large source and package downloads |

The validated host had 62 GiB RAM and a separate larger Docker/containerd data
area. That made the full build practical.

## Before Building

```bash
make audit
docker system df
df -h
```

If the root disk is small, move Docker's data-root or containerd storage before
running `make build`.

## Reducing Pressure

Build in stages:

```bash
make build-base
make build-rocblas
make build-pytorch
make build
```

If PyTorch compilation overwhelms the host, lower the `MAX_JOBS` value in
`docker/Dockerfile`.

## Cleaning

Docker cleanup can remove old layers:

```bash
docker system df
docker builder prune
docker image prune
```

Only prune when you are sure you do not need intermediate layers for faster
rebuilds.
