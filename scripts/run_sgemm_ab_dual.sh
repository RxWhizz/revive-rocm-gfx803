#!/usr/bin/env bash
# Build and run the standalone HIP SGEMM variants on both GPUs simultaneously.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
  ARGS=(--shape 512 --shape 1024 --shape 2048 --shape 4096 --repeats 20 --warmup 5)
fi

quote_cmd() {
  printf '%q ' "$@"
}

RUN0=(/workspace/results/sgemm_gfx803_hip --csv /workspace/results/sgemm_ab_gpu0.csv "${ARGS[@]}")
RUN1=(/workspace/results/sgemm_gfx803_hip --csv /workspace/results/sgemm_ab_gpu1.csv "${ARGS[@]}")
CMD0="bash scripts/build_sgemm_kernels.sh && $(quote_cmd "${RUN0[@]}")"
CMD1="bash scripts/build_sgemm_kernels.sh && $(quote_cmd "${RUN1[@]}")"

echo "GPU 0 -> HIP SGEMM variants"
HIP_VISIBLE_DEVICES=0 bash "$REPO_DIR/scripts/run_container.sh" bash -lc "$CMD0" \
  > "$REPO_DIR/logs/sgemm_ab_gpu0.log" 2>&1 &
P0=$!

echo "GPU 1 -> HIP SGEMM variants"
HIP_VISIBLE_DEVICES=1 bash "$REPO_DIR/scripts/run_container.sh" bash -lc "$CMD1" \
  > "$REPO_DIR/logs/sgemm_ab_gpu1.log" 2>&1 &
P1=$!

wait "$P0"; S0=$?
wait "$P1"; S1=$?

echo "GPU0 exit=$S0 (logs/sgemm_ab_gpu0.log) | GPU1 exit=$S1 (logs/sgemm_ab_gpu1.log)"
echo "csv: results/sgemm_ab_gpu0.csv results/sgemm_ab_gpu1.csv"
exit $(( S0 != 0 || S1 != 0 ))
