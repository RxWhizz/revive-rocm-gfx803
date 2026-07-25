#!/usr/bin/env bash
# Healthcheck: PyTorch importa y ve >=1 GPU gfx803.
/opt/conda/bin/python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
    print("HIP:", torch.version.hip, "| GPUs:", torch.cuda.device_count())
    sys.exit(0 if ok else 1)
except Exception as e:
    print("healthcheck error:", e); sys.exit(1)
PY
