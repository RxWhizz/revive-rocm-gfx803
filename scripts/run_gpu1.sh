#!/usr/bin/env bash
# Fase 9 — job en GPU 1. Uso: run_gpu1.sh [CMD...] (default: optimizer)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIP_VISIBLE_DEVICES=1 bash "$REPO_DIR/scripts/run_container.sh" "${@:-python tests/test_optimizer.py}"
