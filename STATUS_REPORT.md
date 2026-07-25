# STATUS_REPORT — Revive ROCm gfx803 (2× RX 570)

_Última actualización: 2026-06-29 · Fase actual: **0→2 (gate bloqueado)**_

## 1. Hardware confirmado (auditoría Fase 0)

| Ítem | Valor |
|---|---|
| GPUs | **2× AMD Ellesmere/Polaris [1002:67df]** en PCI `04:00.0` y `05:00.0` → **gfx803** |
| Driver | `amdgpu` cargado (host) |
| Cómputo | `/dev/kfd` presente |
| Render nodes | `renderD128`, `renderD129` (2 ✓) |
| OS / kernel | Ubuntu **24.04.4** · kernel **6.17** |
| RAM / CPU | **62 GiB** · **88 hilos** (mejor que el spec: 16 GB / 8 cores) |
| NVIDIA | ausente |

→ El hardware objetivo **SÍ está en esta máquina**. El proyecto es ejecutable aquí.

## 2. Gate Fase 0 — **BLOQUEADO** (requiere acción del usuario)

| Check | Estado |
|---|---|
| amdgpu cargado | ✅ |
| /dev/kfd | ✅ |
| ≥2 render nodes | ✅ |
| **Docker instalado** | ❌ |
| **usuario en `render`** | ❌ |
| **usuario en `video`** | ❌ |
| **usuario en `docker`** | ❌ |

Usuario actual en: `adm cdrom sudo dip plugdev users lpadmin`.

### Acción requerida (no se ejecuta automáticamente — cambia el host + necesita re-login)
```bash
# 1) Instalar Docker (ver scripts/audit_host.sh para el bloque completo)
# 2) Grupos:
sudo usermod -aG render,video,docker "$USER"
#    → CERRAR SESIÓN y volver a entrar (los grupos no aplican hasta re-login).
# 3) Re-ejecutar el gate:
bash scripts/audit_host.sh
```

## 3. Riesgo central del proyecto: gfx803 fuera de soporte oficial

**gfx803 (Polaris) fue retirado de ROCm tras ~ROCm 4.5.** ROCm 5.x/6.x **no** trae kernels gfx803 en
rocBLAS/Tensile ni en los wheels oficiales de PyTorch. Implicaciones:

- Los wheels oficiales de PyTorch-ROCm darán `no kernel image is available` en gfx803.
- Hay que **compilar PyTorch desde fuente** con `PYTORCH_ROCM_ARCH=gfx803`, y posiblemente regenerar
  kernels Tensile de rocBLAS para gfx803.
- `hipBLASLt` no soporta Polaris → `TORCH_BLAS_PREFER_HIPBLASLT=0` obligatorio.

### Riesgo secundario: ABI KFD vs kernel 6.17
El host (kernel 6.17) provee el amdgpu/KFD; el contenedor provee ROCm user-space. ROCm 5.7 user-space fue
validado contra kernels ~5.15-6.2. Con kernel **6.17** puede haber **mismatch de ABI KFD** (síntoma:
`hipErrorNoDevice` / `device_count()==0` pese a /dev/kfd). Mitigaciones a probar (documentadas, no por defecto):
`HSA_OVERRIDE_GFX_VERSION=8.0.3`, o subir a ROCm 6.1 user-space.

## 4. Estrategia de versiones (Fase 2 — propuesta, ver `environment/versions.env`)

| Componente | Elección | Razón |
|---|---|---|
| Base Docker | **Ubuntu 22.04** | ROCm 5.7 targetea 22.04 (no 24.04) |
| ROCm | **5.7.1** user-space | última con ruta gfx803 comunitaria viable; rocBLAS clásico (no hipBLASLt) |
| Python | 3.10 | compat ROCm 5.7 + MACE |
| PyTorch | **2.2.2 compilado desde fuente** (`PYTORCH_ROCM_ARCH=gfx803`) | wheels oficiales no traen gfx803 |
| NumPy | <2.0 (1.26.x) | compat torch 2.2 / MACE |
| ASE | fijada | — |
| MACE | mace-torch compatible torch 2.2 | Fase 8 |

Plan B si ROCm 5.7 no habla con kernel 6.17: probar imagen base ROCm 6.1 (`rocm/dev-ubuntu-22.04:6.1`) +
gfx803 vía `PYTORCH_ROCM_ARCH` + `HSA_OVERRIDE_GFX_VERSION`.

## 5. Qué funcionó / qué falló (se irá llenando)

- ✅ Auditoría host; hardware gfx803 ×2 detectado.
- ⏸️ Build bloqueado por gate (Docker + grupos).
- ⬜ Pendiente: imagen Docker, compilación PyTorch gfx803, pruebas FP32, estabilidad 30 min, MACE.

## 6. Siguiente paso
1. **Usuario**: instala Docker + grupos + re-login (sección 2).
2. Luego: Fase 2 (lockfile) → Fase 3-5 (Dockerfile multi-stage + compilación PyTorch gfx803) → Fase 6 pruebas.

## 7. Hitos de validación (2026-06-29)

- **Gate PASA**: Docker 29.6.1 + grupos render/video/docker. Data-root: imagen ext4 loop 200G en "Nuevo vol"
  (NTFS), montada en /var/lib/containerd vía bind (Docker 29 usa el image-store de containerd → las imágenes
  viven en /var/lib/containerd, NO en /var/lib/docker). El SSD raíz / es de 109G (~12G libres).
- **✅ ABI KFD funciona**: rocminfo en `rocm/dev-ubuntu-22.04:5.7.1` detecta **2× gfx803 "Radeon RX 570 Series"**
  con kernel 6.17, SIN errores HSA, SIN necesitar HSA_OVERRIDE. (Era el mayor riesgo del proyecto.)
- **⚠️ rocBLAS sin gfx803**: `/opt/rocm/lib/rocblas/library/` no tiene Tensile gfx803 (0 archivos). El matmul
  fallaría. → hay que rebuild rocBLAS para gfx803 (Tensile) ADEMÁS de PyTorch desde fuente.

### Decisión pendiente (Fase 5): cómo obtener kernels gfx803 de rocBLAS
- **(A) rocBLAS source/reference para gfx803** (Tensile sin asm-kernels): build rápido (~1h), matmul correcto
  pero más lento. Alineado con "FP32 + correctitud primero". Reproducible.
- **(B) rocBLAS Tensile completo gfx803**: build largo (varias horas, miles de kernels asm), rendimiento
  óptimo. Reproducible pero costoso.
- **(C) Artefactos gfx803 de comunidad** (wheels/libs prebuilt, pin SHA256): rápido, menos "puro".

## 8. RESULTADOS DE VALIDACIÓN (2026-06-29) — ✅ STACK gfx803 FUNCIONA

Imagen final: `revive-pytorch-gfx803:rocm5.7.1-torch2.2.2-py310`. PyTorch `2.2.0a0+git39901f2` (fuente, gfx803).

| Prueba | GPU 0 (RX570 4G) | GPU 1 (RX570 8G) | Resultado |
|---|---|---|---|
| Detección (torch ve 2× gfx803) | ✓ | ✓ | **OK** (HIP 5.7, 32 CUs c/u) |
| Matmul FP32 (=CPU) | ✓ | ✓ | **OK** (~1.25 TFLOPS @ 4096; correcto) |
| Backward (autograd) | ✓ | — | **OK** (grads finitos) |
| Adam (loss baja) | ✓ | ✓ | **OK** (0.63→1e-5) |
| Aislamiento HIP_VISIBLE_DEVICES | ✓ | ✓ | **OK** (1 GPU c/u) |
| Dual simultáneo (2 procesos) | ✓ | ✓ | **OK** (exit 0 ambos) |
| MACE import | — | — | **OK** (torch+ase+mace 0.3.6) |
| MACE inferencia (E+fuerzas) | ✓ | — | **OK** (E=-126.8eV, finito) |
| MACE entrenamiento mínimo | ✓ | — | **OK** (3 épocas, 4 ckpts; req. --E0s average) |
| Estrés 30 min (dual) | ✓ 286k it | ✓ 285k it | **OK** (0 NaN, 0 crashes) |

### Bugs de Dockerfile resueltos (gfx803/ROCm packaging)
1. `rocm-llvm-dev` inexistente → quitado (la imagen dev ya trae LLVM/hipcc).
2. `install.sh -d` apt sin `apt-get update` (base borró las listas) → añadido update.
3. conda primero en PATH rompía joblib/loky de Tensile → PATH sin conda en stage rocblas.
4. CMake 4.x rechaza `cmake_minimum_required<3.5` de submódulos PyTorch 2.2 → pin `cmake==3.26.4`.
5. Imagen dev sin libs ROCm (rocrand/miopen/…) que PyTorch busca → apt install + `apt-mark hold rocblas` (protege gfx803).

### Notas técnicas clave (reproducir / extender)
- rocBLAS gfx803: `release/rocm-rel-5.7` + `./install.sh -d -a gfx803 -i` → 72 kernels Tensile gfx803.
- PyTorch: v2.2.2 fuente, `PYTORCH_ROCM_ARCH=gfx803`, `TORCH_BLAS_PREFER_HIPBLASLT=0`; ccache acelera re-builds.
- `apt-mark hold rocblas rocblas-dev` IMPRESCINDIBLE al instalar otras libs ROCm (si no, hipblas-dev pisa el gfx803).
- ABI ROCm 5.7 ↔ kernel 6.17 OK sin `HSA_OVERRIDE`.
- MIOpen 5.7 no trae kdb gfx803 (convoluciones harían JIT/fallarían) — MACE usa e3nn (matmul/scatter), no afecta.

## 9. Criterios de aceptación — estado
TODOS los criterios CUMPLIDOS (estrés 30min: 286k/285k iters, 0 NaN). Stack reconstruible vía `make build` (Dockerfile + lockfile).
Extensión a gfx900/906/908 (Vega/MI50/MI100): añadir a `PYTORCH_ROCM_ARCH` y rebuild — MÁS fácil que gfx803
(esas SÍ tienen soporte ROCm moderno).
