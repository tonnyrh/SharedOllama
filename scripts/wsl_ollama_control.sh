#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
shift || true

HOST="127.0.0.1"
PORT="11434"
STATE_DIR="${HOME}/.sharedollama"
PID_FILE="${STATE_DIR}/ollama.pid"
LOG_FILE="${STATE_DIR}/ollama.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-127.0.0.1}"
      shift 2
      ;;
    --port)
      PORT="${2:-11434}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$STATE_DIR"

json() {
  local ok="$1"
  local action="$2"
  local running="$3"
  local pid="$4"
  local note="$5"
  printf '{"ok":%s,"action":"%s","running":%s,"pid":"%s","host":"%s","port":"%s","log_file":"%s","message":"%s"}\n' \
    "$ok" "$action" "$running" "$pid" "$HOST" "$PORT" "$LOG_FILE" "$note"
}

if ! command -v ollama >/dev/null 2>&1; then
  json "false" "$ACTION" "false" "" "ollama command not found"
  exit 1
fi

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "$pid"
      return 0
    fi
  fi
  local pid
  pid="$(pgrep -f "ollama serve" | head -n 1 || true)"
  if [[ -n "$pid" ]]; then
    echo "$pid"
    return 0
  fi
  return 1
}

start_ollama() {
  if pid="$(is_running)"; then
    json "true" "start" "true" "$pid" "ollama already running"
    return 0
  fi

  (
    export OLLAMA_HOST="${HOST}:${PORT}"
    nohup ollama serve >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  )

  sleep 1
  if pid="$(is_running)"; then
    json "true" "start" "true" "$pid" "ollama started"
    return 0
  fi

  json "false" "start" "false" "" "failed to start ollama"
  return 1
}

stop_ollama() {
  local stopped="false"
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      stopped="true"
    fi
    rm -f "$PID_FILE"
  fi

  pkill -f "ollama serve" >/dev/null 2>&1 || true

  if pid="$(is_running)"; then
    json "false" "stop" "true" "$pid" "ollama is still running"
    return 1
  fi

  if [[ "$stopped" == "true" ]]; then
    json "true" "stop" "false" "" "ollama stopped"
  else
    json "true" "stop" "false" "" "ollama was not running"
  fi
}

case "$ACTION" in
  start)
    start_ollama
    ;;
  stop)
    stop_ollama
    ;;
  restart)
    stop_ollama >/dev/null
    start_ollama
    ;;
  status)
    if pid="$(is_running)"; then
      json "true" "status" "true" "$pid" "ollama running"
    else
      json "true" "status" "false" "" "ollama not running"
    fi
    ;;
  *)
    json "false" "$ACTION" "false" "" "invalid action"
    exit 2
    ;;
esac
