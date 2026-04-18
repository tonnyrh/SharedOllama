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
- Request totals and rate-limit metrics
- Logs, errors, and alerts

API endpoints:

- `GET /monitor/api/state` full monitor state payload
- `GET /monitor/api/queue` queued request details
- `GET /monitor/api/models` current loaded models

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
