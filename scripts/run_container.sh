#!/usr/bin/env bash
# Wrapper de `docker run` con acceso a GPU gfx803 (flags del spec). Uso: run_container.sh [CMD...]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/environment/versions.env"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

DOCKER="docker"
# si el usuario no está en el grupo docker en esta sesión, usar sg
if ! docker info >/dev/null 2>&1; then DOCKER="sg docker -c"; fi

run() {
  local tty_args=(--interactive)
  if [ -t 1 ]; then tty_args+=(--tty); fi
  local args=(run --rm "${tty_args[@]}"
    --device=/dev/kfd --device=/dev/dri
    --group-add video --group-add render
    --ipc=host --security-opt seccomp=unconfined
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
    -v "$REPO_DIR/tests:/workspace/tests"
    -v "$REPO_DIR/mace:/workspace/mace"
    -v "$REPO_DIR/scripts:/workspace/scripts"
    -v "$REPO_DIR/benchmarks:/workspace/benchmarks"
    -v "$REPO_DIR/docs:/workspace/docs"
    -v "$REPO_DIR/logs:/workspace/logs"
    -v "$REPO_DIR/results:/workspace/results"
    -v "$REPO_DIR/checkpoints:/workspace/checkpoints"
    "$IMAGE" "$@")
  if [ "$DOCKER" = "docker" ]; then docker "${args[@]}"; else sg docker -c "docker $(printf '%q ' "${args[@]}")"; fi
}
run "$@"
