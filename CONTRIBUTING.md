# Contributing

Reports from additional Polaris / gfx803 cards are the most useful
contribution.

Please include:

```bash
make audit
make detect
make test
```

Also include:

- GPU model and VRAM size.
- Device id from `lspci -nn`.
- Linux distribution and kernel.
- Docker version.
- Whether `HSA_OVERRIDE_GFX_VERSION=8.0.3` was required.
- Exact failing command and the last 80 log lines if something failed.

Pull requests should keep the default path conservative: FP32 first,
correctness before speed, Docker isolation, and one process per GPU.
