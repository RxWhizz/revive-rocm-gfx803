#!/usr/bin/env bash
# Fase 0 — Auditoría del host para 2× AMD RX 570 (gfx803) + ROCm/PyTorch.
# No destructivo: solo lee estado. Escribe logs/host_audit.txt y evalúa el gate.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO_DIR/logs/host_audit.txt"
mkdir -p "$REPO_DIR/logs"
exec > >(tee "$LOG") 2>&1

sec() { printf '\n================ %s ================\n' "$1"; }

echo "Auditoría host gfx803 — $(date -u +%Y-%m-%dT%H:%M:%SZ)"

sec "Sistema"
uname -a
(lsb_release -a 2>/dev/null) || { . /etc/os-release && echo "$PRETTY_NAME"; }
echo "cmdline: $(cat /proc/cmdline)"

sec "Memoria / disco"
free -h
df -h / "$REPO_DIR"

sec "GPUs AMD (lspci -nnk)"
lspci -nnk 2>/dev/null | grep -A3 -iE "VGA|3D controller|Display"

sec "Driver amdgpu / amdkfd"
lsmod 2>/dev/null | grep -E "^amdgpu|^amdkfd" || echo "AVISO: amdgpu/amdkfd NO en lsmod"

sec "Nodos de cómputo y render"
ls -l /dev/kfd 2>/dev/null || echo "AVISO: /dev/kfd NO existe"
ls -l /dev/dri/ 2>/dev/null
RENDER_NODES=$(ls /dev/dri/ 2>/dev/null | grep -c '^renderD')
echo "render nodes: $RENDER_NODES"

sec "Detalle por GPU (PCI / driver / render / PCIe / BAR / device-id)"
for pci in $(lspci -nn 2>/dev/null | grep -iE "VGA|3D controller" | grep -i "AMD/ATI" | awk '{print $1}'); do
  echo "--- GPU $pci ---"
  lspci -nnks "$pci" 2>/dev/null | sed 's/^/  /'
  echo "  device-id: $(lspci -nns "$pci" | grep -oE '\[1002:[0-9a-f]{4}\]' | head -1)"
  echo "  PCIe link: $(lspci -vvs "$pci" 2>/dev/null | grep -i 'LnkSta:' | head -1 | sed 's/^[[:space:]]*//')"
  echo "  arquitectura esperada: gfx803 (Polaris/Ellesmere) si device-id ∈ {67df,67ef,6fdf,...}"
done

sec "Grupos del usuario"
echo "usuario: $USER"
groups

sec "Docker"
if command -v docker >/dev/null 2>&1; then
  docker version 2>&1 | head -20
  echo "--- docker info (resumen) ---"
  docker info 2>&1 | grep -iE "Server Version|Storage Driver|Cgroup|Runtimes|Root Dir" || true
else
  echo "AVISO: docker NO instalado"
fi

sec "dmesg amdgpu/kfd (requiere permisos; puede estar vacío)"
(dmesg 2>/dev/null | grep -iE "amdgpu|kfd" | tail -25) || echo "(dmesg no accesible sin sudo)"

# ---------------- Evaluación del gate ----------------
sec "EVALUACIÓN DEL GATE (Fase 0)"
FAIL=0
chk() { if eval "$2"; then echo "  [OK]   $1"; else echo "  [FALLA] $1"; FAIL=1; fi; }

chk "amdgpu cargado"                 'lsmod 2>/dev/null | grep -q "^amdgpu"'
chk "/dev/kfd existe"                'test -e /dev/kfd'
chk ">=2 render nodes"               '[ "$(ls /dev/dri/ 2>/dev/null | grep -c "^renderD")" -ge 2 ]'
chk "Docker instalado"               'command -v docker >/dev/null 2>&1'
chk "usuario en grupo render"        'id -nG | tr " " "\n" | grep -qx render'
chk "usuario en grupo video"         'id -nG | tr " " "\n" | grep -qx video'
chk "usuario en grupo docker"        'id -nG | tr " " "\n" | grep -qx docker'

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "GATE: PASA — se puede continuar a Fase 2 (versiones) / build."
else
  echo "GATE: BLOQUEADO — corrige lo anterior antes de continuar. Comandos correctivos:"
  command -v docker >/dev/null 2>&1 || cat <<'EOF'
  # Instalar Docker (Ubuntu 24.04):
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
EOF
  echo "  # Agregar usuario a grupos (luego CERRAR SESIÓN y volver a entrar):"
  echo "  sudo usermod -aG render,video,docker \"\$USER\""
  echo "  # NO se reinicia ni cierra sesión automáticamente. Re-loguea para aplicar los grupos."
fi
echo ""
echo "Log guardado en: $LOG"
exit "$FAIL"
