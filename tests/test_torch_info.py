#!/usr/bin/env python3
"""Fase 6 — info de PyTorch/HIP. Sale != 0 si no detecta >=2 GPU."""
import sys
import torch


def main() -> int:
    print("PyTorch:", torch.__version__)
    print("HIP:", torch.version.hip)
    print("Disponible:", torch.cuda.is_available())
    n = torch.cuda.device_count()
    print("GPU detectadas:", n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {torch.cuda.get_device_name(i)} | arch={getattr(p,'gcnArchName',getattr(p,'name','?'))} "
              f"| VRAM={p.total_memory/1024**3:.1f} GiB | CUs={getattr(p,'multi_processor_count','?')}")
    ok = torch.cuda.is_available() and n >= 2
    print("RESULTADO:", "OK (2 GPU)" if ok else "FALLA (se esperaban 2 GPU)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
