#!/usr/bin/env bash
# Fase 9 — job en GPU 0. Uso: run_gpu0.sh [CMD...] (default: optimizer)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIP_VISIBLE_DEVICES=0 bash "$REPO_DIR/scripts/run_container.sh" "${@:-python tests/test_optimizer.py}"
