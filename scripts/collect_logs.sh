#!/usr/bin/env bash
# Recolecta logs/resultados y refresca la tabla de resultados en STATUS_REPORT.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
echo "=== logs ==="; ls -la logs/ 2>/dev/null
echo "=== results ==="; ls -la results/ 2>/dev/null
echo "=== resúmenes JSON ==="
for j in results/*.json results/*_summary.json; do
  [ -f "$j" ] && { echo "--- $j ---"; cat "$j"; echo; }
done
echo "=== imágenes docker ==="
(docker images 2>/dev/null || sg docker -c "docker images") | grep -E "revive-pytorch-gfx803|REPOSITORY" || true
