#!/usr/bin/env bash
# Fase 9 — dos procesos simultáneos, uno por GPU (no DDP, no sumar VRAM). Sale !=0 si alguno falla.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-python tests/test_optimizer.py}"

echo "GPU 0 → $CMD"
HIP_VISIBLE_DEVICES=0 bash "$REPO_DIR/scripts/run_container.sh" bash -lc "$CMD" \
  > "$REPO_DIR/logs/gpu0_dual.log" 2>&1 &
P0=$!
echo "GPU 1 → $CMD"
HIP_VISIBLE_DEVICES=1 bash "$REPO_DIR/scripts/run_container.sh" bash -lc "$CMD" \
  > "$REPO_DIR/logs/gpu1_dual.log" 2>&1 &
P1=$!

wait "$P0"; S0=$?
wait "$P1"; S1=$?
echo "GPU0 exit=$S0 (logs/gpu0_dual.log) | GPU1 exit=$S1 (logs/gpu1_dual.log)"
exit $(( S0 != 0 || S1 != 0 ))
