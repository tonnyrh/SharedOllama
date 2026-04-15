# SharedOllama

Shared Ollama server for multiple local Docker projects and optional remote clients.

## Start

```powershell
docker compose --project-name sharedollama up -d
```

## Verify

```powershell
docker compose --project-name sharedollama ps
Invoke-WebRequest -UseBasicParsing http://localhost:11435/api/version
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

## Stop

```powershell
docker compose --project-name sharedollama down
```

## Firewall (Windows, remote access)

```powershell
New-NetFirewallRule -DisplayName "SharedOllama 11435" -Direction Inbound -Protocol TCP -LocalPort 11435 -Action Allow
```
