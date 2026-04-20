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

Minimal usage example:

- Input: `POST /api/generate` with headers `x-real-ip: 192.168.1.62` and `x-client-name: remote-n8n`
- Output: monitor shows client IP `192.168.1.62` and client label `remote-n8n | 192.168.1.62`

API endpoints:

- `GET /monitor/api/state` full monitor state payload
- `GET /monitor/api/queue` queued request details
- `GET /monitor/api/clients` client summary and control state
- `POST /monitor/api/clients/{client_key}/state` set `pause`, `block`, or `resume`
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

For remote clients:

```text
OLLAMA_URL=http://<HOST_IP>:11435
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
