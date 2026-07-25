#!/usr/bin/env python3
"""Extract real GEMM shapes from ROCBLAS_LAYER=2 logs.

Example:
  ROCBLAS_LAYER=2 python mace/test_mace_inference.py 2> results/gemm_shapes.log
  python scripts/parse_rocblas_shapes.py results/gemm_shapes.log --csv results/gemm_shapes.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


INT_KEYS = ("m", "n", "k", "lda", "ldb", "ldc", "stride_a", "stride_b", "stride_c", "batch_count")
TRANS_RE = re.compile(r"\b(trans[ab])\s*[:=]\s*([A-Za-z_]+)", re.IGNORECASE)
INT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(-?\d+)", re.IGNORECASE)


def normalize_trans(value: str) -> str:
    value = value.lower()
    if value in {"n", "none", "rocblas_operation_none"}:
        return "N"
    if value in {"t", "transpose", "rocblas_operation_transpose"}:
        return "T"
    if value in {"c", "conjugate_transpose", "rocblas_operation_conjugate_transpose"}:
        return "C"
    return value.upper()


def parse_line(line: str) -> dict[str, str | int] | None:
    if "rocblas_" not in line.lower() or "gemm" not in line.lower():
        return None
    lower = line.lower()
    precision = "unknown"
    if "rocblas_sgemm" in lower or "_sgemm" in lower:
        precision = "sgemm"
    elif "rocblas_dgemm" in lower or "_dgemm" in lower:
        precision = "dgemm"
    elif "rocblas_hgemm" in lower or "_hgemm" in lower:
        precision = "hgemm"

    rec: dict[str, str | int] = {"precision": precision, "transa": "?", "transb": "?"}
    for key, value in TRANS_RE.findall(line):
        rec[key.lower()] = normalize_trans(value)
    for key, value in INT_RE.findall(line):
        key = key.lower()
        if key in INT_KEYS:
            rec[key] = int(value)

    if not all(key in rec for key in ("m", "n", "k")):
        return None
    if "batch_count" not in rec:
        rec["batch_count"] = 1
    return rec


def shape_key(rec: dict[str, str | int]) -> tuple:
    return (
        rec.get("precision", "unknown"),
        rec.get("transa", "?"),
        rec.get("transb", "?"),
        int(rec["m"]),
        int(rec["n"]),
        int(rec["k"]),
        int(rec.get("batch_count", 1)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"missing log: {args.log}", file=sys.stderr)
        return 2

    counts: Counter[tuple] = Counter()
    for line in args.log.read_text(errors="replace").splitlines():
        rec = parse_line(line)
        if rec is not None:
            counts[shape_key(rec)] += 1

    rows = [
        {
            "count": count,
            "precision": key[0],
            "transa": key[1],
            "transb": key[2],
            "m": key[3],
            "n": key[4],
            "k": key[5],
            "batch_count": key[6],
        }
        for key, count in counts.most_common()
    ]

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["count", "precision", "transa", "transb", "m", "n", "k", "batch_count"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    print("count,precision,transa,transb,m,n,k,batch_count")
    for row in rows[: args.top]:
        print(
            f"{row['count']},{row['precision']},{row['transa']},{row['transb']},"
            f"{row['m']},{row['n']},{row['k']},{row['batch_count']}"
        )
    if not rows:
        print("No rocblas_*gemm calls found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
