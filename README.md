# SharedOllama

SharedOllama is a WSL-native proxy and monitor for Ollama.
It gives you one stable endpoint for clients, plus queueing, rate limiting, client controls, and an admin UI.

## Architecture

Traffic flow:

1. Proxy API listens on port `11434`.
2. Proxy forwards to backend Ollama at `http://127.0.0.1:11435`.
3. Admin UI and admin API listen on port `11444`.

Default URLs:

- Proxy API: `http://localhost:11434`
- Admin UI: `http://localhost:11444/monitor`
- Admin API state: `http://localhost:11444/monitor/api/admin/state`

## Unified Setup (Windows + WSL)

Run this from Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_wsl.ps1
```

What it does:

- Writes runtime config for single WSL Ollama instance.
- Configures Windows firewall rules (`11434`, `11444`) when run as Administrator.
- Configures Windows portproxy to WSL proxy/admin ports when run as Administrator.
- Calls WSL installer and starts `sharedollama-proxy` and `sharedollama-admin` services.
- Verifies proxy and admin endpoints.

Useful flags:

- `-SkipOllamaInstall`
- `-SkipFirewall`
- `-SkipPortProxy`
- `-SkipWslInstall`
- `-Distro <name>`
- `-UseMirroredNetworking` — writes `~/.wslconfig` with `networkingMode=mirrored`. Requires Windows build `22621.2359` or later. Run `wsl --shutdown` after, then rerun setup.

Minimal usage example:

- Input: `powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_wsl.ps1 -SkipOllamaInstall`
- Output: `Service started successfully.`

## WSL-only Install

Run inside your WSL distro:

```bash
cd /path/to/SharedOllama
chmod +x scripts/install_wsl.sh scripts/wsl_ollama_control.sh
./scripts/install_wsl.sh
```

Installer behavior:

- Installs required packages (`python3`, `venv`, `pip`, `curl`, `jq`, `lsof`).
- Optionally installs Ollama.
- Creates venv in `~/.sharedollama/.venv`.
- Installs monitor dependencies.
- Creates `monitor/runtime_config.json` if missing.
- Configures and starts user services:
  - `sharedollama-proxy.service` (port `11434`)
  - `sharedollama-admin.service` (port `11444`)

## Start And Verify

Check service status:

```bash
systemctl --user status sharedollama-proxy.service
systemctl --user status sharedollama-admin.service
```

Quick endpoint checks:

```bash
curl -s http://localhost:11434/api/version
curl -s http://localhost:11434/api/tags
curl -s http://localhost:11444/monitor/api/admin/state
```

## Client Configuration

Use this endpoint in clients:

- Local: `http://localhost:11434`
- Docker on same host: `http://host.docker.internal:11434`
- LAN clients: `http://HOST_IP:11434`

Optional identity and priority headers:

- `x-client-name`
- `x-client-id`
- `x-client-key`
- `x-client-priority` (`0..1000`, lower is higher priority)

## API Endpoints

Monitor/API:

- `GET /monitor/api/state`
- `GET /monitor/api/queue`
- `GET /monitor/api/clients`
- `POST /monitor/api/clients/{client_key}/state`
- `GET /monitor/api/history`
- `GET /monitor/api/history/{request_id}`

Admin:

- `GET /monitor/api/admin/state`
- `POST /monitor/api/admin/config`
- `POST /monitor/api/admin/ollama/{action}` where action is `start|stop|restart|status`

## Security

Set token to require auth for monitor/admin APIs:

```bash
export MONITOR_TOKEN="change-me"
```

## Remote Client Identity

With `netsh portproxy` (NAT mode), all LAN clients appear as `localhost` or a Windows internal IP in the monitor.
This is a Windows networking limitation — portproxy terminates the TCP connection and opens a new one, discarding the original source IP.

Workarounds:

1. **Upgrade Windows** to build `22621.2359` or later and use `-UseMirroredNetworking`. WSL will share the host IP and no portproxy is needed.
2. **Set a client identity header** in your client app:
   ```
   x-client-name: my-laptop
   ```
   SharedOllama will use this as the display name in monitor regardless of IP.

To check your Windows build:
```powershell
[System.Environment]::OSVersion.Version
# Or: winver
```
Minimum required build for mirrored networking: `22621.2359` (Windows 11 22H2, October 2023 update).

Current status for older builds (for example `22621.525`):

- Keep `netsh portproxy` enabled for `11434` and `11444`.
- Expect monitor to show internal relay IPs (for example `172.18.x.x`) instead of true LAN source IP.
- Use `x-client-name` (or `x-client-id`) to identify remote clients reliably until Windows is updated.

## Troubleshooting

- WSL control log: `~/.sharedollama/ollama.log`
- If LAN clients cannot connect to `11434`, run setup script in elevated PowerShell to apply firewall and portproxy.
- If corporate policy blocks local firewall rules, request GPO inbound allow for TCP `11434` and `11444`.
