#!/usr/bin/env bash
# Capture rocBLAS GEMM calls from a workload and summarize the real shapes.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

WORKLOAD="${1:-python mace/test_mace_inference.py}"
GPU="${GPU:-0}"
LOG_NAME="${LOG_NAME:-gemm_shapes_gpu${GPU}.log}"
LOG="$REPO_DIR/results/$LOG_NAME"
CSV="${LOG%.log}.csv"

echo "GPU=$GPU"
echo "workload: $WORKLOAD"
echo "raw log: $LOG"

HIP_VISIBLE_DEVICES="$GPU" bash "$REPO_DIR/scripts/run_container.sh" bash -lc \
  "ROCBLAS_LAYER=2 $WORKLOAD" \
  > "$REPO_DIR/logs/capture_gemm_shapes_gpu${GPU}.stdout.log" \
  2> "$LOG"
STATUS=$?

python3 "$REPO_DIR/scripts/parse_rocblas_shapes.py" "$LOG" --csv "$CSV" \
  > "$REPO_DIR/results/gemm_shapes_gpu${GPU}_summary.txt" 2>&1
PARSE_STATUS=$?

echo "workload exit=$STATUS | parser exit=$PARSE_STATUS"
echo "csv: $CSV"
echo "summary: results/gemm_shapes_gpu${GPU}_summary.txt"
exit $(( STATUS != 0 || PARSE_STATUS != 0 ))
