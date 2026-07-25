#!/usr/bin/env bash
# Entrypoint: activa el entorno conda y exporta el env gfx803, luego ejecuta el comando.
set -e
export PATH=/opt/rocm/bin:/opt/conda/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
# Activar micromamba env si existe el python del prefix
if [ -x /opt/conda/bin/python ]; then
  export PATH=/opt/conda/bin:$PATH
fi
exec "$@"
