# Publication Checklist

Before changing the repository visibility to public:

- Confirm the default branch is `main`.
- Confirm `README.md` opens with the public quickstart.
- Confirm `LICENSE` is present.
- Confirm large raw logs, benchmark traces, and checkpoints are not tracked.
- Confirm release `v0.1.0` has the validation archive and checksum file.
- Set the GitHub description:

```text
Revive AMD Polaris/gfx803 GPUs for small FP32 AI workloads with ROCm, PyTorch, Docker, and MACE.
```

- Add topics:

```text
rocm, gfx803, polaris, rx570, rx580, pytorch, mace, machine-learning, amd-gpu, docker
```

- Enable Issues so users can submit GPU reports.
- Keep the repository public claim honest: tested on RX 570, expected on similar
  Polaris cards, unsupported by AMD.
