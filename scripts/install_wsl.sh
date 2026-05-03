#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${HOME}/.sharedollama"
VENV_DIR="${STATE_ROOT}/.venv"
SERVICE_DIR="${HOME}/.config/systemd/user"
PROXY_SERVICE="${SERVICE_DIR}/sharedollama-proxy.service"
ADMIN_SERVICE="${SERVICE_DIR}/sharedollama-admin.service"
RUNTIME_CONFIG="${ROOT_DIR}/monitor/runtime_config.json"

INSTALL_OLLAMA="true"
ENABLE_SERVICE="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-ollama-install)
      INSTALL_OLLAMA="false"
      shift
      ;;
    --skip-service)
      ENABLE_SERVICE="false"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

log() {
  echo "[sharedollama-install] $1"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

log "Installing required system packages"
require_command sudo
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl jq lsof

if [[ "$INSTALL_OLLAMA" == "true" ]] && ! command -v ollama >/dev/null 2>&1; then
  log "Installing Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
fi

log "Creating Python virtual environment"
mkdir -p "$STATE_ROOT"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "${ROOT_DIR}/monitor/requirements.txt"

log "Preparing runtime configuration"
if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  cat >"$RUNTIME_CONFIG" <<'JSON'
{
  "backend_url": "http://127.0.0.1:11434",
  "shared_port": 11435,
  "ollama_host": "127.0.0.1",
  "ollama_port": 11434,
  "updated_at": ""
}
JSON
fi

chmod +x "${ROOT_DIR}/scripts/wsl_ollama_control.sh"

if [[ "$ENABLE_SERVICE" == "true" ]]; then
  log "Configuring user systemd services"
  mkdir -p "$SERVICE_DIR"
  
  # Proxy service (11434)
  cat >"$PROXY_SERVICE" <<EOF
[Unit]
Description=SharedOllama Proxy
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}/monitor
Environment=SHAREDOLLAMA_RUNTIME_CONFIG=${ROOT_DIR}/monitor/runtime_config.json
ExecStart=${VENV_DIR}/bin/python ${ROOT_DIR}/monitor/app.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

  # Admin service (11444)
  cat >"$ADMIN_SERVICE" <<EOF
[Unit]
Description=SharedOllama Admin
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}/monitor
Environment=SHAREDOLLAMA_RUNTIME_CONFIG=${ROOT_DIR}/monitor/runtime_config.json
Environment=SHAREDOLLAMA_OLLAMA_CONTROL_SCRIPT=${ROOT_DIR}/scripts/wsl_ollama_control.sh
ExecStart=${VENV_DIR}/bin/python ${ROOT_DIR}/monitor/admin.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

  if systemctl --user daemon-reload >/dev/null 2>&1; then
    systemctl --user daemon-reload
    systemctl --user enable --now sharedollama-proxy.service
    systemctl --user enable --now sharedollama-admin.service
    log "Systemd user services started"
    log "  - sharedollama-proxy.service (port 11434)"
    log "  - sharedollama-admin.service (port 11444)"
  else
    log "Systemd user mode is not available in this WSL distro"
    log "Start manually:"
    log "  Proxy: ${VENV_DIR}/bin/python ${ROOT_DIR}/monitor/app.py"
    log "  Admin: ${VENV_DIR}/bin/python ${ROOT_DIR}/monitor/admin.py"
  fi
fi

log "Installation complete"
log "Input: Start the service"
log "Output: Service started successfully."
