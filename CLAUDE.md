# SharedOllama — AI Assistant Instructions

SharedOllama is a WSL-native HTTP proxy and admin monitor for Ollama.
It provides a single stable endpoint for LAN/Docker clients with rate limiting,
priority queueing, client controls, and an admin web UI.

## Architecture

```
Clients (apps, LAN, Docker)
         ↓ port 11434
   Proxy  (monitor/app.py)     FastAPI — forwards to Ollama, runs the queue
         ↓
   Ollama backend               port 11435 (default)

Admin UI / API (monitor/admin.py)   port 11444 — browser dashboard, proxies state from proxy
```

Key files:
- [monitor/app.py](monitor/app.py) — proxy, queue workers, rate limiting, client identity
- [monitor/admin.py](monitor/admin.py) — admin UI + API, inline HTML/JS dashboard
- [monitor/shared.py](monitor/shared.py) — `MonitorState`: all shared state, config, model cache
- [monitor/runtime_config.json](monitor/runtime_config.json) — persisted backend URL / ports (git-ignored)
- [scripts/install_wsl.sh](scripts/install_wsl.sh) — WSL installer (venv, systemd services)
- [scripts/setup_windows_wsl.ps1](scripts/setup_windows_wsl.ps1) — Windows setup (firewall, portproxy, scheduled task)

## Queue behaviour

Non-streaming requests (`stream: false`) go through an `asyncio.PriorityQueue`.
Background workers (default: 2) dequeue and forward to Ollama one at a time.
Streaming requests (`stream: true` or omitted) bypass the queue and go direct.
Priority range: 0 (highest) – 1000 (lowest). Set via `x-client-priority` header.

## Testing locally

```bash
curl http://localhost:11434/health
curl http://localhost:11434/api/tags
curl http://localhost:11444/monitor/api/admin/state
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","prompt":"hello","stream":false}'
```

## Services

```bash
systemctl --user status sharedollama-proxy.service
systemctl --user status sharedollama-admin.service
systemctl --user restart sharedollama-proxy.service
```

## Local Ollama Worker skill

This project ships a Claude Code skill at [skills/ollama-worker/](skills/ollama-worker/).
Install it to delegate bounded coding tasks to local Qwen Coder and save cloud tokens.

```powershell
# Install
.\scripts\install_skill.ps1

# Remove
.\scripts\install_skill.ps1 -Uninstall
```

The skill routes requests with `stream: false` and `x-client-priority: 0` through the
SharedOllama proxy, giving code assistance the highest queue priority.
Invoke with: `/ollama-worker <task>`

To disable without uninstalling: set `DISABLE_OLLAMA_WORKER=1` in your environment.
Claude Code will skip suggesting the skill when this variable is present.
