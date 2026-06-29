# SharedOllama — Codex Agent Instructions

SharedOllama is a WSL-native HTTP proxy and admin monitor for Ollama.
It provides a single stable endpoint with rate limiting, priority queueing,
client controls, and a web admin UI.

## Architecture

```
Clients → Proxy (port 11434, monitor/app.py) → Ollama backend (port 11435)
           Admin UI (port 11444, monitor/admin.py) ← browser dashboard
```

Key files:
- `monitor/app.py` — proxy, queue workers, rate limiting, client identity
- `monitor/admin.py` — admin UI + API, proxies state from proxy process
- `monitor/shared.py` — MonitorState: all shared state, config, model cache

## Queue behaviour

- `stream: false` requests → priority queue → processed by background workers
- `stream: true` (or omitted) → direct to Ollama, rate limited only
- Priority: 0 (highest) – 1000 (lowest), set via `x-client-priority` header

## Local AI assistance

A local Qwen Coder model runs via SharedOllama at `http://localhost:11434`.
For bounded coding tasks, prefer the local model to reduce cloud token usage.

Call the helper script directly:

```bash
python skills/ollama-worker/scripts/call_ollama.py \
  --model qwen2.5-coder:7b \
  --system "You are a focused coding assistant." \
  --user "<task + minimal context>"
```

Or inline:

```python
import urllib.request, json

payload = json.dumps({
    "model": "qwen2.5-coder:7b",
    "stream": False,
    "messages": [
        {"role": "system", "content": "You are a focused coding assistant."},
        {"role": "user",   "content": "<task + minimal context>"}
    ]
}).encode()

req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=payload,
    headers={"Content-Type": "application/json", "x-client-priority": "0"},
)
with urllib.request.urlopen(req, timeout=120) as r:
    content = json.loads(r.read())["message"]["content"]
```

Always use `stream: false` and `x-client-priority: 0` for code assistance requests.

Use the local model for:
- Implementing one function or method
- Modifying one file
- Generating tests
- Small scripts, regex, documentation

Skip the local model for:
- Architecture decisions
- Security or compliance analysis
- Multi-file refactors
- Unclear requirements

If `DISABLE_OLLAMA_WORKER=1` is set, use cloud assistance directly.

## Testing

```bash
curl http://localhost:11434/health
curl http://localhost:11434/api/tags
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder:7b","prompt":"hello","stream":false}'
```

## Services

```bash
systemctl --user restart sharedollama-proxy.service
systemctl --user restart sharedollama-admin.service
```

## Coding standards

- Simple, readable, maintainable — no over-engineering
- Comments only when the WHY is non-obvious
- All comments and user-facing text in English
- Never commit, push, or delete files without explicit instruction
