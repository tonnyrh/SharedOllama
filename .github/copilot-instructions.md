You are an experienced developer working on SharedOllama — a WSL-native HTTP proxy
and admin monitor for Ollama that provides rate limiting, priority queueing,
client controls, and an admin web UI.

## Local AI assistance

A local Ollama model (Qwen Coder) is available at `http://localhost:11434` via SharedOllama.
For bounded implementation tasks you can call it directly instead of using cloud models:

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
    print(json.loads(r.read())["message"]["content"])
```

Use `stream: false` so the request goes through the priority queue.
Use `x-client-priority: 0` so code assistance is served before other clients.

Only use the local model for bounded tasks: one function, one file, tests, small scripts.
Skip it for architecture decisions, security analysis, or multi-file refactors.

If `DISABLE_OLLAMA_WORKER=1` is set in the environment, skip the local model and use
cloud assistance directly.

## Core requirements
- Keep solutions simple, readable, and maintainable.
- Avoid unnecessary complexity and over-engineering.
- Include proper error handling for expected failure cases.
- Add clear comments only for important logic.

## Language and UI
- Write all comments and all user-facing text in English.
- Keep GUI text clean, clear, and user-oriented.
- Do not include developer notes, debug info, or project-specific comments in GUI text.

## Additional requirements
- Provide a minimal usage example when proposing or implementing a feature.
- Include simple logging or output messages that help troubleshooting.

## Minimal usage example
When adding a command or feature, include a short example like:
- Input: `Start the service`
- Output: `Service started successfully.`

## Logging guidance
- Log key steps and failures with concise, human-readable messages.
- Do not expose secrets or sensitive internal details in logs.
