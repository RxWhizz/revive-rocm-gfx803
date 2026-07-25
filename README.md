# revive-rocm-gfx803 — PyTorch + ROCm para 2× AMD RX 570 (gfx803)

Entorno reproducible, aislado y reversible (Docker + micromamba) para usar **dos RX 570 Polaris (gfx803)**
con PyTorch; primera app científica: **MACE**. ALIGNN/DGL en una segunda fase.

> **Estado**: ver [STATUS_REPORT.md](STATUS_REPORT.md). Hardware 2×RX570 confirmado en el host;
> **gate Fase 0 bloqueado** (falta Docker + grupos render/video/docker).

## 1. Propósito
Revivir GPUs Polaris (sin soporte oficial en ROCm moderno) para inferencia y entrenamiento pequeño en FP32,
un proceso independiente por GPU. Sin FP16/BF16/torch.compile/Triton/DDP/DGL al inicio.

## 2. Hardware
2× RX 570 (gfx803, [1002:67df]) · Ubuntu 24.04 · kernel 6.17 · 62 GiB RAM · 88 hilos.

## 3. Arquitectura (aislamiento obligatorio)
```
Host Ubuntu (kernel, amdgpu, /dev/kfd, /dev/dri, Docker)
└── Docker (ROCm 5.7 user-space, HIP, rocBLAS, toolchain, parches gfx803)
    └── micromamba (Python 3.10, PyTorch-ROCm gfx803, NumPy, ASE, MACE)
```
El host NO instala PyTorch/MACE. Ver `environment/versions.env`.

## 4. Por qué se compila PyTorch desde fuente
gfx803 fue retirado de ROCm tras 4.5 → los wheels oficiales dan `no kernel image is available`. Se compila
con `PYTORCH_ROCM_ARCH=gfx803` y `TORCH_BLAS_PREFER_HIPBLASLT=0` (ver `configs/gfx803.env`).

## 5. Uso (Makefile envuelve scripts reproducibles)
```bash
make audit       # Fase 0 — auditoría host + gate   (scripts/audit_host.sh)
make build       # Fase 5 — construir imagen         (tras pasar el gate)
make shell       # entrar al contenedor
make test        # Fase 6 — torch info/matmul/backward/adam
make test-dual   # dos procesos, uno por GPU
make benchmark   # Fase 7 — estrés 30 min
make matmul-ab   # A/B torch.matmul/rocBLAS FP32 en ambas GPUs
make sgemm-ab    # A/B kernels HIP propios: conservador/LDS/float4
make gemm-shapes # capturar formas GEMM reales vía ROCBLAS_LAYER=2
make mace-test   # Fase 8 — import/inferencia/entrenamiento mínimo
make report      # recolecta logs → STATUS_REPORT
```

Guía de optimización GEMM/gfx803: [docs/gemm_gfx803_optimization.md](docs/gemm_gfx803_optimization.md).

## 6. Configuración del host (gate)
Instalar Docker + `sudo usermod -aG render,video,docker $USER` + **re-login**. Comandos completos en
`scripts/audit_host.sh` (sección correctiva) y STATUS_REPORT §2.

## 7-11. Build · Pruebas · MACE · Ejecución por GPU · Diagnóstico
Por fases (ver el .md de especificación y STATUS_REPORT). Diagnóstico cubre `no kernel image`, hipBLASLt,
OOM, fallo de una GPU, fallo de micromamba.

## 12. Rollback
Todo vive en Docker + este repo. Rollback = borrar imagen/contenedor; el host queda intacto (solo se le
añadió Docker + grupos, reversibles). Ningún paquete científico se instala en el host.

## 13. Limitaciones conocidas
- gfx803 sin soporte oficial: rendimiento y estabilidad no garantizados; FP16/BF16 evitados.
- Posible mismatch ABI KFD (ROCm 5.7 user-space ↔ kernel 6.17); mitigación `HSA_OVERRIDE_GFX_VERSION=8.0.3`.
- No se suma VRAM de las 2 GPU; paralelismo = 1 proceso/GPU.

## Extensión futura (gfx900/906/908)
Añadir el target a `PYTORCH_ROCM_ARCH` (p.ej. `gfx803;gfx900;gfx906`) y reconstruir. Vega/MI50/MI100 SÍ
tienen soporte ROCm moderno → más fácil que gfx803.

### Tabla de resultados (se llena al ejecutar pruebas)
| Prueba | GPU 0 | GPU 1 | Resultado |
|---|---|---|---|
| Detección | | | ⬜ |
| Matmul FP32 | | | ⬜ |
| Backward | | | ⬜ |
| Adam | | | ⬜ |
| Estrés 30 min | | | ⬜ |
| MACE import | | | ⬜ |
| MACE inferencia | | | ⬜ |
| MACE entrenamiento mínimo | | | ⬜ |
