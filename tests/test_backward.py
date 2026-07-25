#!/usr/bin/env python3
"""Fase 6 — red 128→512→512→1: forward, MSE, backward, gradientes válidos (sin NaN/Inf)."""
import sys
import torch
import torch.nn as nn

DEV = "cuda"


def main() -> int:
    if not torch.cuda.is_available():
        print("FALLA: cuda no disponible"); return 1
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(128, 512), nn.SiLU(),
                        nn.Linear(512, 512), nn.SiLU(),
                        nn.Linear(512, 1)).to(DEV)
    x = torch.randn(64, 128, device=DEV)
    y = torch.randn(64, 1, device=DEV)
    pred = net(x)
    loss = nn.functional.mse_loss(pred, y)
    print(f"  forward loss = {loss.item():.4f}")
    loss.backward()
    bad = 0
    for name, p in net.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            print(f"  grad inválido en {name}"); bad += 1
    ok = torch.isfinite(loss).all() and bad == 0
    print("RESULTADO:", "OK (forward+backward, grads finitos)" if ok else "FALLA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
