#!/usr/bin/env python3
"""Fase 7 — estrés/estabilidad. `--minutes N`: bucle de entrenamiento + registro de métricas, checkpoints,
NaN, errores, temperatura (rocm-smi). Ante fallo guarda traceback y conserva resultados previos."""
import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

DEV = "cuda"
RESULTS = Path("/workspace/results"); RESULTS.mkdir(parents=True, exist_ok=True)
CKPT = Path("/workspace/checkpoints"); CKPT.mkdir(parents=True, exist_ok=True)


def gpu_temp() -> str:
    try:
        out = subprocess.run(["rocm-smi", "--showtemp"], capture_output=True, text=True, timeout=10).stdout
        temps = [l.split(":")[-1].strip() for l in out.splitlines() if "Temperature" in l and "edge" in l.lower()]
        return ";".join(temps) if temps else "n/a"
    except Exception:
        return "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        print("FALLA: cuda no disponible"); return 1

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(args.size, args.size), nn.SiLU(),
                        nn.Linear(args.size, args.size), nn.SiLU(),
                        nn.Linear(args.size, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, foreach=False, fused=False)
    x = torch.randn(256, args.size, device=DEV)
    y = torch.randn(256, 1, device=DEV)

    log = RESULTS / f"benchmark_gpu{gpu}.jsonl"
    t_end = time.time() + args.minutes * 60
    it = 0; nans = 0
    print(f"[gpu {gpu}] estrés {args.minutes} min, size {args.size}")
    try:
        with open(log, "w") as f:
            while time.time() < t_end:
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(net(x), y)
                loss.backward(); opt.step()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                lv = loss.item()
                if not torch.isfinite(loss):
                    nans += 1
                rec = {"it": it, "t": round(time.time(), 1), "loss": round(lv, 6),
                       "iter_s": round(dt, 4), "mem_MB": round(torch.cuda.memory_allocated()/1024**2, 1),
                       "temp": gpu_temp() if it % 200 == 0 else None}
                f.write(json.dumps(rec) + "\n")
                if it % 500 == 0:
                    f.flush()
                    torch.save(net.state_dict(), CKPT / f"bench_gpu{gpu}.pt")
                    print(f"  [gpu {gpu}] it {it} loss {lv:.5f} {dt*1000:.1f}ms/it temp {rec['temp']}")
                it += 1
        ok = nans == 0
        print(f"[gpu {gpu}] FIN: {it} iters, {nans} NaN | RESULTADO:", "OK" if ok else "FALLA (NaN)")
        json.dump({"gpu": gpu, "iters": it, "nans": nans, "minutes": args.minutes},
                  open(RESULTS / f"benchmark_gpu{gpu}_summary.json", "w"), indent=2)
        return 0 if ok else 1
    except Exception:
        (RESULTS / f"benchmark_gpu{gpu}_traceback.txt").write_text(traceback.format_exc())
        print(f"[gpu {gpu}] EXCEPCIÓN en it {it} (traceback guardado)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
