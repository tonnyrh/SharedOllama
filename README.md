# SharedOllama

Shared Ollama server for multiple local Docker projects and optional remote clients.
Now includes a built-in monitor/proxy layer for queueing, request limiting, and live operational visibility.

## Start

```powershell
docker compose --project-name sharedollama up -d
```

## Verify

```powershell
docker compose --project-name sharedollama ps
Invoke-WebRequest -UseBasicParsing http://localhost:11435/api/version
Invoke-WebRequest -UseBasicParsing http://localhost:11435/monitor/summary
```

## Monitor page

Open:

```text
http://localhost:11435/monitor
```

If `MONITOR_TOKEN` is set, open with:

```text
http://localhost:11435/monitor?token=<MONITOR_TOKEN>
```

This page provides:

- Loaded model visibility
- Queue depth and queue item details
- Client identification in queue and history, including host/IP and best-effort client hints
- Pause, stop/block, and resume controls per client
- Request totals and rate-limit metrics
- Live graph for total requests, queue size, and completed requests
- Logs, errors, and alerts

Client identification priority in monitor:

- `x-real-ip`, `true-client-ip`, `cf-connecting-ip`, `x-client-ip`
- then first IP in `x-forwarded-for`
- then `forwarded` header (`for=...`)
- fallback to socket peer IP

Optional stable client label:

- Send `x-client-name: my-service-name` or `x-client-id: my-service-id`
- Without these headers, monitor falls back to `user-agent` + resolved client IP
- For strict per-client control behind Docker/NAT, send `x-client-key: my-stable-client-key`
- Optional request priority header: `x-client-priority: 0..1000` (lower value = higher priority)

Minimal usage example:

- Input: `POST /api/generate` with headers `x-real-ip: 192.168.1.62` and `x-client-name: remote-n8n`
- Output: monitor shows client IP `192.168.1.62` and client label `remote-n8n | 192.168.1.62`

Priority usage example:

- Input: `POST /api/chat` with headers `x-client-key: nightly-batch` and `x-client-priority: 300`
- Output: requests for `nightly-batch` are queued behind clients with priority values below `300`

API endpoints:

- `GET /monitor/api/state` full monitor state payload
- `GET /monitor/api/queue` queued request details
- `GET /monitor/api/clients` client summary and control state
- `POST /monitor/api/clients/{client_key}/state` set `pause`, `block`, or `resume`
- `POST /monitor/api/clients/{client_key}/priority` set client queue priority (`0..1000`)
- `GET /monitor/api/models` current loaded models
- `GET /monitor/graph` simple live graph view

## Pull a model

```powershell
docker exec shared-ollama ollama pull qwen2.5:7b
```

## Client configuration

For Dockerized clients on the same host:

```text
OLLAMA_URL=http://host.docker.internal:11435
```

Set a stable identity per client process:

```text
X_CLIENT_KEY=my-client-name
X_CLIENT_PRIORITY=100
```

For remote clients:

```text
OLLAMA_URL=http://<HOST_IP>:11435
```

## Windows and WSL port strategy

If `localhost:11434` is already occupied by another Windows service or relay, keep this proxy on `11435` and point clients explicitly to that port.

Example for Ollama CLI on Windows:

```powershell
set OLLAMA_HOST=http://127.0.0.1:11435
ollama list
```

Optional override when the monitor should target another backend (for example WSL or host service):

```text
OLLAMA_BACKEND_URL=http://ollama:11434
```

## Queue and rate-limit settings

Configure in `.env`:

```text
RATE_LIMIT_PER_MINUTE=120
MAX_QUEUE_SIZE=200
QUEUE_WORKERS=2
ALERT_QUEUE_THRESHOLD=50
UPSTREAM_TIMEOUT_SECONDS=300
MONITOR_TOKEN=
```

## Stop

```powershell
docker compose --project-name sharedollama down
```

## Firewall (Windows, remote access)

```powershell
New-NetFirewallRule -DisplayName "SharedOllama 11435" -Direction Inbound -Protocol TCP -LocalPort 11435 -Action Allow
```
