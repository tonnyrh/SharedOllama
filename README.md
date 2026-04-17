# SharedOllama

Shared Ollama server for multiple local Docker projects and optional remote clients.
Includes a monitoring proxy that records traffic and usage telemetry.

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

## Monitoring data captured

The proxy records:

- Incoming traffic (all proxied requests)
- Credits used (prompt + completion token counts when available)
- Remote IP and remote location headers (if available)
- Model used
- Memory used (from `/api/ps` model memory fields)
- Response time
- Errors/status codes
- Prompt
- Answer

## Monitoring endpoints

- `GET /monitor/summary` - aggregated metrics
- `GET /monitor/requests?limit=100` - recent request logs

## Retention (automatic delete)

Logs are automatically deleted after 30 days by default.
You can configure this with:

```text
MONITOR_RETENTION_DAYS=30
```

## Stop

```powershell
docker compose --project-name sharedollama down
```

## Firewall (Windows, remote access)

```powershell
New-NetFirewallRule -DisplayName "SharedOllama 11435" -Direction Inbound -Protocol TCP -LocalPort 11435 -Action Allow
```
