#!/usr/bin/env python3
"""Fase 8 — MACE importa (torch + ase + mace)."""
import sys


def main() -> int:
    try:
        import torch, ase, mace
        print("torch", torch.__version__, "| HIP", torch.version.hip,
              "| ase", ase.__version__, "| mace", getattr(mace, "__version__", "?"))
        print("RESULTADO: OK")
        return 0
    except Exception as e:
        print("FALLA import:", e); return 1


if __name__ == "__main__":
    sys.exit(main())
