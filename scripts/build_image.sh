#!/usr/bin/env bash
# Fase 5 — build reproducible. Construye por etapas, registra logs/hashes, etiqueta con versiones.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source environment/versions.env
mkdir -p logs/build results

DK() { if docker info >/dev/null 2>&1; then docker "$@"; else sg docker -c "docker $(printf '%q ' "$@")"; fi; }

TARGET="${1:-runtime}"        # base | rocblas | pytorch | runtime
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
LOG="logs/build/build_${TARGET}_$(date +%Y%m%d_%H%M%S).log"

echo "==> build target=$TARGET  imagen=$IMAGE"
echo "    commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)  fecha: $(date -u +%FT%TZ)"
df -h "$REPO_DIR" | awk 'NR==2{print "    disco repo libre: "$4}'

# RAM 62G aquí; PyTorch MAX_JOBS configurable
export DOCKER_BUILDKIT=1
DK build --target "$TARGET" \
  -f docker/Dockerfile \
  -t "${IMAGE_NAME}:${TARGET}" \
  $( [ "$TARGET" = "runtime" ] && echo -t "$IMAGE" ) \
  --build-arg MAMBA_VERSION="${MAMBA_VERSION}" \
  . 2>&1 | tee "$LOG"

echo "==> hashes de la imagen"
DK image inspect "${IMAGE_NAME}:${TARGET}" --format '{{.Id}}' | tee -a "$LOG"
echo "==> prueba de importación rápida"
[ "$TARGET" = "runtime" ] && bash scripts/run_container.sh python tests/test_torch_info.py || true
echo "OK. log: $LOG"
