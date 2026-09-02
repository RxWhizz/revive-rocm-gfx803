#!/usr/bin/env bash
# Phase 0 - host audit for AMD Polaris / gfx803 + ROCm/PyTorch.
# Non-destructive: reads host state, writes logs/host_audit.txt, evaluates gate.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO_DIR/logs/host_audit.txt"
mkdir -p "$REPO_DIR/logs"
exec > >(tee "$LOG") 2>&1

sec() { printf '\n================ %s ================\n' "$1"; }

echo "gfx803 host audit - $(date -u +%Y-%m-%dT%H:%M:%SZ)"

sec "System"
uname -a
(lsb_release -a 2>/dev/null) || { . /etc/os-release && echo "$PRETTY_NAME"; }
echo "cmdline: $(cat /proc/cmdline)"

sec "Memory / disk"
free -h
df -h / "$REPO_DIR"

sec "AMD GPUs (lspci -nnk)"
lspci -nnk 2>/dev/null | grep -A3 -iE "VGA|3D controller|Display"

sec "amdgpu / amdkfd driver"
lsmod 2>/dev/null | grep -E "^amdgpu|^amdkfd" || echo "WARNING: amdgpu/amdkfd not present in lsmod"

sec "Compute and render nodes"
ls -l /dev/kfd 2>/dev/null || echo "WARNING: /dev/kfd does not exist"
ls -l /dev/dri/ 2>/dev/null
RENDER_NODES=$(ls /dev/dri/ 2>/dev/null | grep -c '^renderD')
echo "render nodes: $RENDER_NODES"

sec "Per-GPU detail (PCI / driver / render / PCIe / BAR / device id)"
for pci in $(lspci -nn 2>/dev/null | grep -iE "VGA|3D controller" | grep -i "AMD/ATI" | awk '{print $1}'); do
  echo "--- GPU $pci ---"
  lspci -nnks "$pci" 2>/dev/null | sed 's/^/  /'
  echo "  device-id: $(lspci -nns "$pci" | grep -oE '\[1002:[0-9a-f]{4}\]' | head -1)"
  echo "  PCIe link: $(lspci -vvs "$pci" 2>/dev/null | grep -i 'LnkSta:' | head -1 | sed 's/^[[:space:]]*//')"
  echo "  expected architecture: gfx803 (Polaris/Ellesmere) for device ids such as 67df, 67ef, 6fdf"
done

sec "User groups"
echo "user: $USER"
groups

sec "Docker"
if command -v docker >/dev/null 2>&1; then
  docker version 2>&1 | head -20
  echo "--- docker info (summary) ---"
  docker info 2>&1 | grep -iE "Server Version|Storage Driver|Cgroup|Runtimes|Root Dir" || true
else
  echo "WARNING: docker is not installed"
fi

sec "dmesg amdgpu/kfd (requires permission; may be empty)"
(dmesg 2>/dev/null | grep -iE "amdgpu|kfd" | tail -25) || echo "(dmesg not accessible without sudo)"

# ---------------- Evaluación del gate ----------------
sec "GATE EVALUATION (Phase 0)"
FAIL=0
chk() { if eval "$2"; then echo "  [OK]   $1"; else echo "  [FAIL] $1"; FAIL=1; fi; }

chk "amdgpu loaded"                  'lsmod 2>/dev/null | grep -q "^amdgpu"'
chk "/dev/kfd exists"                'test -e /dev/kfd'
chk ">=2 render nodes"               '[ "$(ls /dev/dri/ 2>/dev/null | grep -c "^renderD")" -ge 2 ]'
chk "Docker installed"               'command -v docker >/dev/null 2>&1'
chk "user in render group"           'id -nG | tr " " "\n" | grep -qx render'
chk "user in video group"            'id -nG | tr " " "\n" | grep -qx video'
chk "user in docker group"           'id -nG | tr " " "\n" | grep -qx docker'

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "GATE: PASS - continue to version/build phases."
else
  echo "GATE: BLOCKED - fix the failed checks before building. Corrective commands:"
  command -v docker >/dev/null 2>&1 || cat <<'EOF'
  # Install Docker (Ubuntu 24.04):
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
EOF
  echo "  # Add your user to groups, then log out completely and log back in:"
  echo "  sudo usermod -aG render,video,docker \"\$USER\""
  echo "  # Group changes do not apply to already-open shells."
fi
echo ""
echo "Log saved at: $LOG"
exit "$FAIL"
