#!/usr/bin/env python3
"""Summarize JSON files produced by tests/bench_matmul_ab.py."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    rows = []
    for raw in sys.argv[1:]:
        path = Path(raw)
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, list):
                rows.extend(data)
    if not rows:
        print("No matmul_ab JSON rows found.")
        return 1

    print("case,gpu,m,n,k,avg_ms,tflops,temp_c,power_w,ok")
    by_shape = defaultdict(float)
    for row in sorted(rows, key=lambda x: (str(x.get("gpu_visible")), x["m"], x["n"], x["k"])):
        ok = bool(row.get("finite")) and (row.get("matches_cpu_sample") is not False)
        key = (int(row["m"]), int(row["n"]), int(row["k"]))
        by_shape[key] += float(row["tflops"])
        print(
            f"{row.get('case')},{row.get('gpu_visible')},{row['m']},{row['n']},{row['k']},"
            f"{row['avg_ms']},{row['tflops']},{row.get('temp_c')},{row.get('power_w')},{ok}"
        )

    if len({str(row.get("gpu_visible")) for row in rows}) > 1:
        print("")
        print("aggregate_tflops_by_shape")
        for shape, tflops in sorted(by_shape.items()):
            print(f"{shape[0]}x{shape[1]}x{shape[2]},{tflops:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
