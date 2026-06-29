---
name: ollama-worker
description: Delegate a bounded coding task to a local Ollama model (Qwen Coder). Claude acts as architect and reviewer; the local model handles implementation. Use for modifying one file, implementing one function, refactoring a method, generating tests, small scripts, regex, or documentation. Call with /ollama-worker <task description>.
---

# Local Ollama Worker

Delegate small, well-defined coding tasks to the local Ollama model (Qwen Coder) via SharedOllama. Claude remains architect and reviewer. The local model handles implementation only.

---

## When to use

Qualified tasks — bounded, single concern:

- Modify one file
- Implement one function
- Refactor one method
- Generate tests for a file or function
- Convert between languages
- Small PowerShell or Bash scripts
- Regex generation
- Documentation generation

Do NOT use for:

- Multiple subsystems
- Architecture decisions
- Security analysis
- Compliance or legal review
- Large refactors across many files
- Unclear or ambiguous requirements

If the task does not qualify, say so and handle it directly without delegating.

---

## Workflow

### Step 1 — Check model availability

```bash
curl -s http://localhost:11434/api/tags
```

Select model in this order of preference:

1. `qwen3-coder` (any variant)
2. `qwen2.5-coder` (any variant)
3. `qwen2.5` (any variant)
4. Any available model as fallback

If Ollama is unreachable or no model is available, return `LOCAL_MODEL_UNAVAILABLE` and stop.
Do NOT fall back to cloud automatically — Claude decides whether to retry or escalate.

### Step 2 — Gather minimal context

Use Read, Grep, or Glob tools. Read the minimum needed:

- Prefer targeted line ranges over whole-file reads
- Use Grep to locate relevant functions or symbols
- Read at most 2–3 files, only relevant sections

Never read the entire project or unrelated files.

### Step 3 — Call Ollama

Use the helper script:

```powershell
python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\call_ollama.py" `
  --model <selected-model> `
  --system "<concise system prompt>" `
  --user "<task description + only the relevant context>"
```

Or inline with Python if the context needs building dynamically:

```python
import urllib.request, json

payload = json.dumps({
    "model": "<selected>",
    "stream": False,
    "messages": [
        {"role": "system", "content": "<system prompt>"},
        {"role": "user",   "content": "<task + context>"}
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

**Always:**

- `"stream": false` — routes through the SharedOllama priority queue (non-blocking for other clients)
- `"x-client-priority": "0"` — highest priority; code assistance is served before other queued work

### Step 4 — Apply changes

Parse the model response for code blocks. Apply changes with the Edit tool — targeted edits only. Never overwrite whole files. If the change is large or uncertain, generate a diff description first and confirm before applying.

### Step 5 — Run tests

If the project has a known test command, run it and note the result.

### Step 6 — Return structured result

Always end with this JSON block:

```json
{
  "summary": "One sentence describing what was done",
  "files_changed": ["relative/path/to/file.py"],
  "diff": "Short description or unified diff of the change",
  "tests": { "executed": true, "passed": true },
  "warnings": [],
  "confidence": 0.92
}
```

If `confidence` is below 0.70, flag it explicitly and ask Claude to review before applying any patch.

---

## Safety rules

- Never commit, push, or merge
- Never delete files
- Never overwrite a file without a targeted Edit
- Never expose secrets or credentials in prompts sent to Ollama
- If the local model fails or returns garbage: return `LOCAL_MODEL_UNAVAILABLE`, stop, let Claude decide next step

---

## Token optimisation

Return only what Claude needs to review:

- `summary`
- `diff` (not the full file)
- `tests` result
- `warnings`

Never return full file contents unless the user explicitly asks for it.
Claude reviews the summary and diff only. Full context is requested only when confidence is low.

---

## Roles

| Role | Responsibility |
|------|----------------|
| Claude | Architect, reviewer, decision maker |
| Ollama / Qwen | Implementation worker for bounded tasks |
| OpenRouter | Overflow — only when Claude decides to escalate |
