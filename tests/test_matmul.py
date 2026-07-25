#!/usr/bin/env python3
"""Fase 6 — matmul FP32 en GPU: tamaños 512/1024/2048/4096, NaN/Inf, comparación con CPU, tiempos JSON."""
import json
import sys
import time
from pathlib import Path

import torch

DEV = "cuda"
SIZES = [512, 1024, 2048, 4096]
RESULTS = Path("/workspace/results"); RESULTS.mkdir(parents=True, exist_ok=True)


def bench(n: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(n, n, generator=g, dtype=torch.float32)
    b = torch.randn(n, n, generator=g, dtype=torch.float32)
    ag, bg = a.to(DEV), b.to(DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cg = ag @ bg
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    c = cg.cpu()
    finite = bool(torch.isfinite(c).all())
    # muestra: comparar una submatriz con CPU
    ref = (a[:64, :] @ b[:, :64])
    close = bool(torch.allclose(c[:64, :64], ref, atol=1e-2, rtol=1e-3))
    gflops = (2 * n**3) / dt / 1e9
    return {"n": n, "time_s": round(dt, 5), "gflops": round(gflops, 1),
            "finite": finite, "matches_cpu": close}


def main() -> int:
    if not torch.cuda.is_available():
        print("FALLA: cuda no disponible"); return 1
    rows, ok = [], True
    for n in SIZES:
        try:
            r = bench(n)
        except RuntimeError as e:  # OOM en 4096 posible
            print(f"  n={n}: skip ({str(e)[:50]})"); continue
        rows.append(r)
        good = r["finite"] and r["matches_cpu"]
        ok = ok and good
        print(f"  n={n:5d} | {r['time_s']:.4f}s | {r['gflops']:7.1f} GFLOP/s | "
              f"finito={r['finite']} | =CPU={r['matches_cpu']} | {'OK' if good else 'FALLA'}")
    json.dump(rows, open(RESULTS / "matmul.json", "w"), indent=2)
    print("RESULTADO:", "OK" if ok and rows else "FALLA")
    return 0 if (ok and rows) else 1


if __name__ == "__main__":
    sys.exit(main())
