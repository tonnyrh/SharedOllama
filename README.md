# SharedOllama

SharedOllama is now a WSL-native monitor and proxy for Ollama.
It provides request queueing, rate limiting, observability, and a separate admin panel for WSL Ollama lifecycle control.

## Architecture

Traffic flow:

1. **Proxy (port 11434)**: Clients call SharedOllama proxy on standard Ollama port
2. **Backend (port 11434)**: Proxy forwards requests to configured Ollama backend
3. **Admin (port 11444)**: Separate admin service for UI, config, and Ollama control

Default URLs:

- Proxy API: http://localhost:11434 (standard Ollama port)
- Admin UI: http://localhost:11444/monitor
- Live graph: http://localhost:11444/monitor/graph

## Install In WSL

Run inside your WSL distro:

```bash
cd /path/to/SharedOllama
chmod +x scripts/install_wsl.sh scripts/wsl_ollama_control.sh
./scripts/install_wsl.sh
```

What the installer does:

- Installs system packages (python3, venv, pip, curl, jq, lsof)
- Optionally installs Ollama if missing
- Creates virtual environment in ~/.sharedollama/.venv
- Installs monitor Python dependencies
- Creates monitor/runtime_config.json if missing
- Sets up two user systemd services:
  - `sharedollama-proxy` (port 11434)
  - `sharedollama-admin` (port 11444)

Minimal usage example:

- Input: Start the services
- Output: Services started successfully.

## Start And Verify

When systemd user services are available:

```bash
systemctl --user status sharedollama-proxy.service
systemctl --user status sharedollama-admin.service
```

Manual fallback:

```bash
# Terminal 1 - Proxy
source ~/.sharedollama/.venv/bin/activate
python monitor/app.py

# Terminal 2 - Admin
source ~/.sharedollama/.venv/bin/activate
python monitor/admin.py
```

Check endpoints:

```bash
curl -s http://localhost:11434/health
curl -s http://localhost:11434/api/version
curl -s http://localhost:11444/monitor/api/admin/state
```

## Admin Panel (Unified Control)

Open:

http://localhost:11444/monitor

If MONITOR_TOKEN is configured, include it in query:

http://localhost:11444/monitor?token=YOUR_TOKEN

Admin panel supports:

- Manage runtime config (backend URL, Ollama host/port)
- Start WSL Ollama
- Stop WSL Ollama
- Restart WSL Ollama
- Check WSL Ollama status
- Verify backend reachability

Runtime config is persisted in:

- monitor/runtime_config.json

## Client Configuration

Use the SharedOllama proxy URL in all clients:

- Local clients: http://localhost:11434
- Other machines on LAN: http://HOST_IP:11434

Optional stable identity headers:

- x-client-name
- x-client-id
- x-client-key
- x-client-priority (0..1000, lower is higher priority)

## API Endpoints

Core monitor:

- GET /monitor/api/state
- GET /monitor/api/queue
- GET /monitor/api/clients
- POST /monitor/api/clients/{client_key}/state
- GET /monitor/api/history
- GET /monitor/api/history/{request_id}

Admin and setup:

- GET /monitor/api/admin/state
- POST /monitor/api/admin/config
- POST /monitor/api/admin/ollama/{action}

Where action is one of:

- start
- stop
- restart
- status

## Logging And Troubleshooting

SharedOllama logs key events to the monitor logs panel:

- Startup and backend URL
- Queue and processing lifecycle
- Client controls and priority changes
- Admin actions and command outcomes

WSL Ollama control log file:

- ~/.sharedollama/ollama.log

## Security

Set monitor token to require authentication for monitor and admin APIs:

```bash
export MONITOR_TOKEN="change-me"
```

## Windows Firewall (Remote Access)

Run in PowerShell on Windows host:

```powershell
New-NetFirewallRule -DisplayName "SharedOllama 11434" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
netsh interface portproxy add v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11434 connectaddress=127.0.0.1
```

## Install In WSL

Run inside your WSL distro:

```bash
cd /path/to/SharedOllama
chmod +x scripts/install_wsl.sh scripts/wsl_ollama_control.sh
./scripts/install_wsl.sh
```

What the installer does:

- Installs system packages (python3, venv, pip, curl, jq, lsof)
- Optionally installs Ollama if missing
- Creates virtual environment in ~/.sharedollama/.venv
- Installs monitor Python dependencies
- Creates monitor/runtime_config.json if missing
- Sets up user systemd service when available

Minimal usage example:

- Input: Start the service
- Output: Service started successfully.

## Start And Verify

When systemd user services are available:

```bash
systemctl --user status sharedollama-monitor.service
```

Manual fallback:

```bash
source ~/.sharedollama/.venv/bin/activate
python monitor/app.py
```

Check endpoints:

```bash
curl -s http://localhost:11435/health
curl -s http://localhost:11435/api/version
```

## Admin Panel (Unified Control)

Open:

http://localhost:11435/monitor

If MONITOR_TOKEN is configured, include it in query:

http://localhost:11435/monitor?token=YOUR_TOKEN

Admin panel supports:

- Save runtime config (backend URL, proxy port, Ollama host, Ollama port)
- Start WSL Ollama
- Stop WSL Ollama
- Restart WSL Ollama
- Check WSL Ollama status
- Verify backend reachability

Runtime config is persisted in:

- monitor/runtime_config.json

## Client Configuration

Use the SharedOllama proxy URL in all clients:

- Local clients: http://localhost:11435
- Other machines on LAN: http://HOST_IP:11435

Optional stable identity headers:

- x-client-name
- x-client-id
- x-client-key
- x-client-priority (0..1000, lower is higher priority)

## API Endpoints

Core monitor:

- GET /monitor/api/state
- GET /monitor/api/queue
- GET /monitor/api/clients
- POST /monitor/api/clients/{client_key}/state
- POST /monitor/api/clients/{client_key}/priority
- GET /monitor/api/models
- GET /monitor/api/history
- GET /monitor/api/history/{request_id}

Admin and setup:

- GET /monitor/api/admin/state
- POST /monitor/api/admin/config
- POST /monitor/api/admin/ollama/{action}

Where action is one of:

- start
- stop
- restart
- status

## Logging And Troubleshooting

SharedOllama logs key events to the monitor logs panel:

- Startup and backend URL
- Queue and processing lifecycle
- Client controls and priority changes
- Admin actions and command outcomes

WSL Ollama control log file:

- ~/.sharedollama/ollama.log

## Security

Set monitor token to require authentication for monitor and admin APIs:

```bash
export MONITOR_TOKEN="change-me"
```

## Windows Firewall (Remote Access)

Run in PowerShell on Windows host:

```powershell
New-NetFirewallRule -DisplayName "SharedOllama 11435" -Direction Inbound -Protocol TCP -LocalPort 11435 -Action Allow
```
