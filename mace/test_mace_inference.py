#!/usr/bin/env python3
"""Fase 8 — inferencia MACE en GPU: energía + fuerzas finitas sobre una estructura ASE simple. FP32."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

RESULTS = Path("/workspace/results"); RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("FALLA: cuda no disponible"); return 1
    try:
        from mace.calculators import mace_mp
    except Exception as e:
        print("FALLA import mace.calculators:", e); return 1

    # estructura simple: Cu fcc 2x2x2
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    try:
        calc = mace_mp(model="small", dispersion=False, default_dtype="float32", device="cuda")
    except Exception as e:
        print("FALLA cargando modelo MACE:", e); return 1
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    fin = bool(np.isfinite(e) and np.isfinite(f).all())
    fmax = float(np.abs(f).max())
    mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  E = {e:.4f} eV | Fmax = {fmax:.4f} eV/Å | finito={fin} | peak {mem:.0f} MB")
    json.dump({"energy_eV": float(e), "fmax": fmax, "finite": fin, "peak_MB": mem},
              open(RESULTS / "mace_inference.json", "w"), indent=2)
    print("RESULTADO:", "OK" if fin else "FALLA")
    return 0 if fin else 1


if __name__ == "__main__":
    sys.exit(main())
