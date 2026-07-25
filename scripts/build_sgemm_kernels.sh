#!/usr/bin/env bash
# Compile the standalone HIP SGEMM microbenchmark for gfx803.
set -euo pipefail

SRC="${SRC:-/workspace/benchmarks/sgemm_gfx803_hip.cpp}"
OUT="${1:-/workspace/results/sgemm_gfx803_hip}"
HIPCC="${HIPCC:-hipcc}"

mkdir -p "$(dirname "$OUT")"
"$HIPCC" -O3 -std=c++17 --offload-arch=gfx803 "$SRC" -o "$OUT"
echo "built: $OUT"
