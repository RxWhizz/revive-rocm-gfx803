#!/usr/bin/env python3
"""Fase 6 — Adam (foreach=False, fused=False): la pérdida debe bajar. Usado también en ejecución dual."""
import os
import sys
import torch
import torch.nn as nn

DEV = "cuda"


def main() -> int:
    if not torch.cuda.is_available():
        print("FALLA: cuda no disponible"); return 1
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "?")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, foreach=False, fused=False)
    x = torch.randn(256, 64, device=DEV)
    y = (x.sum(1, keepdim=True) * 0.1)
    losses = []
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"  [gpu {gpu}] step {step:3d} loss {loss.item():.5f}")
        losses.append(loss.item())
    dropped = losses[-1] < losses[0]
    finite = all(torch.isfinite(torch.tensor(losses)))
    ok = dropped and finite
    print(f"  [gpu {gpu}] loss {losses[0]:.5f} → {losses[-1]:.5f} | RESULTADO:",
          "OK" if ok else "FALLA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
