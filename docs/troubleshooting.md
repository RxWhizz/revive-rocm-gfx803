# Troubleshooting

Start with the host gate:

```bash
make audit
```

If the gate fails, fix the host before rebuilding the image. Most runtime
errors are downstream of missing KFD, missing render nodes, Docker permissions,
or an image that was rebuilt after rocBLAS was accidentally replaced.

## Docker Is Missing

Symptom:

```text
Docker instalado: FALLA
```

Install Docker on the host, then re-run:

```bash
make audit
```

The audit script prints a conservative Ubuntu install block when Docker is not
found.

## User Is Not In render, video, Or docker

Symptom:

```text
usuario en grupo render: FALLA
usuario en grupo video: FALLA
usuario en grupo docker: FALLA
```

Fix:

```bash
sudo usermod -aG render,video,docker "$USER"
```

Log out completely and log back in. New group membership does not apply to
already-open shells.

## /dev/kfd Is Missing

Symptom:

```text
/dev/kfd existe: FALLA
```

Check that the kernel driver is loaded:

```bash
lsmod | grep -E "^amdgpu|^amdkfd"
lspci -nnk | grep -A3 -iE "VGA|3D controller|Display"
```

If `amdgpu` is not bound to the card, solve the host driver issue first. The
container cannot create KFD support by itself.

## PyTorch Says no kernel image is available

Symptom:

```text
RuntimeError: HIP error: no kernel image is available for execution on the device
```

Likely cause:

- PyTorch was not built for `gfx803`.
- rocBLAS was replaced by a stock package without gfx803 Tensile libraries.

Check:

```bash
make shell
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))
PY
```

Rebuild:

```bash
make build-rocblas
make build-pytorch
make build
```

## torch.cuda.device_count() Is Zero

Symptoms:

```text
Disponible: False
GPU detectadas: 0
```

First verify host devices:

```bash
make audit
```

If host devices exist but PyTorch still sees zero GPUs, try the gfx override
inside `configs/gfx803.env`:

```bash
export HSA_OVERRIDE_GFX_VERSION=8.0.3
```

Use it only as a diagnostic or last resort; the validated RX 570 host did not
need it.

## hipBLASLt Errors

Polaris is not the target for hipBLASLt. Keep:

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

This is already set in the Dockerfile and entrypoint.

## Out Of Memory On 4 GB Cards

Symptoms:

- Matmul at 4096 fails.
- MACE training fails while smaller tests pass.

Try smaller shapes:

```bash
bash scripts/run_matmul_ab_dual.sh --shape 512 --shape 1024 --shape 2048 --repeats 10 --warmup 3
```

For MACE, lower batch size/model size in `mace/test_mace_training.py` or run
only inference.

## Build Runs Out Of Disk

The source build creates large Docker layers. Move Docker storage to a larger
disk or prune old layers:

```bash
docker system df
docker builder prune
docker image prune
```

See [build time and disk planning](build-time-and-disk.md).

## One GPU Works, The Other Fails

Run isolation tests:

```bash
make test-gpu0
make test-gpu1
```

Then inspect:

```bash
rocm-smi
dmesg | grep -iE "amdgpu|kfd|gpu fault"
```

Common causes include unstable risers, power limits, thermal throttling, or a
card with less VRAM than the workload needs.

## What To Attach To An Issue

Attach text, not screenshots:

```bash
make audit
make detect
make test
```

Include:

- `logs/host_audit.txt`
- The exact failing command
- Last 80 lines of the failing build/runtime log
- GPU model, VRAM, distro, kernel, Docker version
