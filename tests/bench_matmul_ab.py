#!/usr/bin/env python3
"""A/B benchmark FP32 GEMM for gfx803 through PyTorch/rocBLAS.

The script intentionally stays in Python so it can run inside the existing
runtime image without rebuilding PyTorch.  Use HIP_VISIBLE_DEVICES outside the
container to pin one process to one physical GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import torch


RESULTS = Path("/workspace/results")
CSV_FIELDS = [
    "case",
    "gpu_visible",
    "device_index",
    "device_name",
    "arch",
    "m",
    "n",
    "k",
    "dtype",
    "avg_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "gflops",
    "tflops",
    "finite",
    "matches_cpu_sample",
    "mem_alloc_mb",
    "temp_c",
    "power_w",
]


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = re.split(r"[x,]", text.lower())
    if len(parts) == 1:
        n = int(parts[0])
        return n, n, n
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; use N or MxNxK")
    return int(parts[0]), int(parts[1]), int(parts[2])


def rocm_smi_snapshot() -> dict[str, float | None]:
    """Best-effort telemetry; missing rocm-smi should not fail the benchmark."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showtemp", "--showpower"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        ).stdout
    except Exception:
        return {"temp_c": None, "power_w": None}

    temp = None
    power = None
    for line in out.splitlines():
        low = line.lower()
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]
        if nums and temp is None and ("temperature" in low or "temp" in low):
            temp = nums[-1]
        if nums and power is None and ("power" in low or "average graphics package power" in low):
            power = nums[-1]
    return {"temp_c": temp, "power_w": power}


def device_info() -> dict[str, str | int | float]:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "arch": getattr(props, "gcnArchName", getattr(props, "name", "?")),
        "total_mem_gb": round(props.total_memory / 1024**3, 3),
        "cu": getattr(props, "multi_processor_count", "?"),
    }


def cpu_sample_ok(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> bool:
    rows = min(32, a.shape[0])
    cols = min(32, b.shape[1])
    ref = a[:rows, :].cpu() @ b[:, :cols].cpu()
    got = c[:rows, :cols].cpu()
    return bool(torch.allclose(got, ref, atol=1e-2, rtol=1e-3))


def bench_shape(
    shape: tuple[int, int, int],
    repeats: int,
    warmup: int,
    seed: int,
    validate: bool,
) -> dict[str, object]:
    m, n, k = shape
    gen = torch.Generator(device="cpu").manual_seed(seed + m * 13 + n * 17 + k * 19)
    a_cpu = torch.randn(m, k, generator=gen, dtype=torch.float32)
    b_cpu = torch.randn(k, n, generator=gen, dtype=torch.float32)
    a = a_cpu.to("cuda")
    b = b_cpu.to("cuda")

    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    times_ms: list[float] = []
    c = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        c = a @ b
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    assert c is not None
    finite = bool(torch.isfinite(c).all())
    matches = cpu_sample_ok(a_cpu, b_cpu, c) if validate else None
    avg_ms = mean(times_ms)
    gflops = (2.0 * m * n * k) / (avg_ms / 1000.0) / 1e9
    telem = rocm_smi_snapshot()

    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": "fp32",
        "avg_ms": round(avg_ms, 4),
        "median_ms": round(median(times_ms), 4),
        "min_ms": round(min(times_ms), 4),
        "max_ms": round(max(times_ms), 4),
        "gflops": round(gflops, 2),
        "tflops": round(gflops / 1000.0, 4),
        "finite": finite,
        "matches_cpu_sample": matches,
        "mem_alloc_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        **telem,
    }


def write_outputs(rows: list[dict[str, object]], out_prefix: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"{out_prefix}.json"
    csv_path = RESULTS / f"{out_prefix}.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path} and {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", action="append", type=parse_shape, default=[],
                    help="Matrix shape as N or MxNxK; can be repeated.")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--case", default="torch_matmul")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FALLA: cuda/HIP no disponible")
        return 1

    shapes = args.shape or [parse_shape(s) for s in ("512", "1024", "2048", "4096")]
    info = device_info()
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "(unset)")
    print(
        f"HIP_VISIBLE_DEVICES={visible} | visible GPUs={torch.cuda.device_count()} | "
        f"device={info['device_name']} | arch={info['arch']} | CUs={info['cu']}"
    )

    rows: list[dict[str, object]] = []
    for shape in shapes:
        rec = bench_shape(shape, args.repeats, args.warmup, args.seed, not args.no_validate)
        row = {
            "case": args.case,
            "gpu_visible": visible,
            "device_index": info["device_index"],
            "device_name": info["device_name"],
            "arch": info["arch"],
            **rec,
        }
        rows.append(row)
        ok = row["finite"] and (row["matches_cpu_sample"] is not False)
        print(
            f"{row['m']}x{row['n']}x{row['k']} | {row['avg_ms']:8.3f} ms | "
            f"{row['tflops']:6.3f} TFLOPS | temp={row['temp_c']}C | "
            f"power={row['power_w']}W | {'OK' if ok else 'FALLA'}"
        )

    suffix = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(visible))
    out_prefix = args.out_prefix or f"matmul_ab_gpu{suffix}"
    write_outputs(rows, out_prefix)
    return 0 if rows and all(r["finite"] and (r["matches_cpu_sample"] is not False) for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
