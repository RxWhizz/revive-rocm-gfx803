#!/usr/bin/env bash
# Run PyTorch/rocBLAS FP32 matmul A/B on both physical GPUs simultaneously.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
  ARGS=(--shape 512 --shape 1024 --shape 2048 --shape 4096 --repeats 30 --warmup 8)
fi

quote_cmd() {
  printf '%q ' "$@"
}

CMD0=(python tests/bench_matmul_ab.py --case torch_rocblas --out-prefix matmul_ab_gpu0 "${ARGS[@]}")
CMD1=(python tests/bench_matmul_ab.py --case torch_rocblas --out-prefix matmul_ab_gpu1 "${ARGS[@]}")

echo "GPU 0 -> ${CMD0[*]}"
HIP_VISIBLE_DEVICES=0 bash "$REPO_DIR/scripts/run_container.sh" bash -lc "$(quote_cmd "${CMD0[@]}")" \
  > "$REPO_DIR/logs/matmul_ab_gpu0.log" 2>&1 &
P0=$!

echo "GPU 1 -> ${CMD1[*]}"
HIP_VISIBLE_DEVICES=1 bash "$REPO_DIR/scripts/run_container.sh" bash -lc "$(quote_cmd "${CMD1[@]}")" \
  > "$REPO_DIR/logs/matmul_ab_gpu1.log" 2>&1 &
P1=$!

wait "$P0"; S0=$?
wait "$P1"; S1=$?

python3 "$REPO_DIR/scripts/summarize_matmul_ab.py" "$REPO_DIR/results"/matmul_ab_gpu*.json \
  > "$REPO_DIR/results/matmul_ab_summary.txt" 2>/dev/null || true

echo "GPU0 exit=$S0 (logs/matmul_ab_gpu0.log) | GPU1 exit=$S1 (logs/matmul_ab_gpu1.log)"
echo "summary: results/matmul_ab_summary.txt"
exit $(( S0 != 0 || S1 != 0 ))
