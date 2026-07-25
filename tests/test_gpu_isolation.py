#!/usr/bin/env python3
"""Fase 6 — aislamiento: con HIP_VISIBLE_DEVICES=N el proceso debe ver EXACTAMENTE 1 GPU."""
import os
import sys
import torch


def main() -> int:
    vis = os.environ.get("HIP_VISIBLE_DEVICES", "(no seteado)")
    n = torch.cuda.device_count()
    print(f"HIP_VISIBLE_DEVICES={vis} | GPUs visibles={n}")
    if n >= 1:
        print(f"  GPU 0 = {torch.cuda.get_device_name(0)}")
        # operación mínima para confirmar que es usable
        x = torch.randn(8, 8, device="cuda"); _ = (x @ x).sum().item()
    ok = (n == 1)
    print("RESULTADO:", "OK (exactamente 1 GPU)" if ok else f"FALLA (se esperaba 1, hay {n})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
