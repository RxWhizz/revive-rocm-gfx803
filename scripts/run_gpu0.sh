#!/usr/bin/env bash
# Phase 9 - run one job on GPU 0. Usage: run_gpu0.sh [CMD...] (default: optimizer)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD=("$@")
if [ "${#CMD[@]}" -eq 0 ]; then
  CMD=(python tests/test_optimizer.py)
fi
HIP_VISIBLE_DEVICES=0 bash "$REPO_DIR/scripts/run_container.sh" "${CMD[@]}"
