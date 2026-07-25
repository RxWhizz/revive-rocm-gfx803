#!/usr/bin/env python3
"""Fase 8 — entrenamiento MACE diminuto: 2-5 épocas, 1 GPU, FP32, batch chico, checkpoint, la pérdida cambia.
Sin DDP. Genera un dataset mínimo (pocas estructuras con energías/fuerzas sintéticas) y entrena un MACE pequeño."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

WORK = Path("/workspace/results/mace_train"); WORK.mkdir(parents=True, exist_ok=True)


def make_dataset(path: Path, n: int = 12):
    """Dataset mínimo: dímeros/trímeros de Cu con energía y fuerzas sintéticas (suaves)."""
    frames = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        d = 2.2 + 0.4 * rng.random()
        a = Atoms("Cu2", positions=[[0, 0, 0], [d, 0, 0]], cell=[10, 10, 10], pbc=True)
        e = float(0.5 * (d - 2.5) ** 2)             # potencial armónico sintético
        fx = -(d - 2.5)
        a.info["REF_energy"] = e
        a.arrays["REF_forces"] = np.array([[fx, 0, 0], [-fx, 0, 0]])
        frames.append(a)
    write(str(path), frames)
    return frames


def main() -> int:
    train = WORK / "train.xyz"
    make_dataset(train)
    cmd = [
        "python", "-m", "mace.cli.run_train",
        "--name", "tiny", "--train_file", str(train), "--valid_fraction", "0.2",
        "--model", "MACE", "--num_interactions", "1", "--num_channels", "8",
        "--max_ell", "1", "--r_max", "4.0", "--batch_size", "2",
        "--max_num_epochs", "3", "--device", "cuda", "--default_dtype", "float32",
        "--E0s", "average",   # E0 atómicas por mínimos cuadrados (requerido por mace 0.3.6)
        "--energy_key", "REF_energy", "--forces_key", "REF_forces",
        "--model_dir", str(WORK), "--checkpoints_dir", str(WORK),
        "--results_dir", str(WORK), "--log_dir", str(WORK), "--seed", "1",
    ]
    print("  lanzando entrenamiento MACE diminuto (3 épocas, 1 GPU, FP32)…")
    r = subprocess.run(cmd, capture_output=True, text=True)
    (WORK / "train_stdout.txt").write_text(r.stdout + "\n=== STDERR ===\n" + r.stderr)
    ckpts = list(WORK.glob("*.model")) + list(WORK.glob("**/*.pt"))
    # señal de éxito: terminó sin error y produjo checkpoint / log de épocas
    epochs = r.stdout.count("Epoch") + r.stderr.count("Epoch")
    ok = (r.returncode == 0) and (len(ckpts) > 0 or epochs > 0)
    print(f"  returncode={r.returncode} | épocas vistas={epochs} | checkpoints={len(ckpts)}")
    json.dump({"returncode": r.returncode, "epochs": epochs, "checkpoints": len(ckpts)},
              open(WORK / "summary.json", "w"), indent=2)
    print("RESULTADO:", "OK" if ok else "FALLA (ver train_stdout.txt)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
