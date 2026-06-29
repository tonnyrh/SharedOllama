---
name: ollama-worker
description: Delegate a bounded coding task to a local Ollama model (Qwen Coder). Claude acts as architect and reviewer; the local model handles implementation. Use for modifying one file, implementing one function, refactoring a method, generating tests, small scripts, regex, or documentation. Call with /ollama-worker <task description>.
---

# Local Ollama Worker

Delegate small, well-defined coding tasks to the local Ollama model (Qwen Coder) via SharedOllama. Claude remains architect and reviewer. The local model handles implementation only.

---

## Task sizing — Claude decides before delegating

Before calling Qwen, estimate whether the task fits the selected model's context window.
The right model changes what's feasible:

| Model | Context | Suitable task size |
|-------|---------|-------------------|
| `qwen2.5-coder:1.5b` | ~8K tokens | Single function, < 100 lines of context |
| `qwen2.5-coder:7b` | ~32K tokens | Single file, up to ~500 lines of context |
| `qwen3-coder` (any) | 32K–128K tokens | Larger files, multi-function rewrites |

**Rule:** If the task + required context exceeds ~60% of the model's context window, split
it into sub-tasks and delegate one at a time. Never truncate context silently — a partial
view produces wrong output.

**Qwen can produce whole files** — use `WRITE_FILE` for new files or complete rewrites.
For targeted edits, prefer `REPLACE_EXACT` or `REPLACE_LINES` to keep output small.

Splitting heuristic:
- Single method/function → one call
- Whole file rewrite (< model limit) → one call with `WRITE_FILE`
- Multiple unrelated changes in one file → split into one call per change
- Changes across multiple files → one call per file, in dependency order

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
- Unclear or ambiguous requirements

If the task does not qualify, say so and handle it directly without delegating.

---

## Workflow

### Step 1 — Check model availability and select

```bash
curl -s http://localhost:11434/api/tags
```

Select model in this order of preference:

1. `qwen3-coder` (any variant) — largest context, best quality
2. `qwen2.5-coder:7b` — good quality, 32K context
3. `qwen2.5-coder:1.5b` — lightweight, small tasks only
4. `qwen2.5` (any variant) — fallback
5. Any available model

Adjust model choice based on task size (see Task sizing above).
If Ollama is unreachable or no model is available, return `LOCAL_MODEL_UNAVAILABLE` and stop.
Do NOT fall back to cloud automatically — Claude decides whether to retry or escalate.

### Step 2 — Gather minimal context

Use Read, Grep, or Glob tools. Read the minimum needed:

- For targeted edits: read only the relevant lines/function
- For whole-file rewrites: read the full file (only if it fits the model)
- Use Grep to locate relevant functions or symbols
- Never read unrelated files

### Step 3 — Call Ollama

**For pure generation tasks** (new function, new file, explain code):

```powershell
python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\call_ollama.py" `
  --model <selected-model> `
  --system "You are a focused coding assistant. Return only the requested code." `
  --user "<task + minimal context>"
```

**For file modification tasks** (edit, replace, inject, whole-file rewrite) — use the
file-op system prompt so Qwen returns a structured FILE_OP block that `apply_op.py`
can execute autonomously:

```powershell
python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\call_ollama.py" `
  --model <selected-model> `
  --system (Get-Content "$env:USERPROFILE\.claude\skills\ollama-worker\prompts\file_op_system.txt" -Raw) `
  --user "<task>\n\nFILE: <path>\n<relevant content>"
```

Always use:
- `"stream": false` — routes through SharedOllama priority queue
- `"x-client-priority": "0"` — highest priority; code assistance served first

### Step 4 — Apply file operations

```powershell
# Pipe directly (apply immediately)
python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\call_ollama.py" ... |
  python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\apply_op.py"

# Dry-run first when confidence is uncertain
python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\call_ollama.py" ... |
  python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\apply_op.py" --dry-run
```

`apply_op.py` exits 0 on success, 1 on parse error, 2 on apply error, and always prints
a JSON result. Stop and report to Claude on any non-zero exit.

### Step 5 — Run tests

If the project has a known test command, run it and note the result.

### Step 6 — Return structured result

Always end with:

```json
{
  "summary": "One sentence describing what was done",
  "files_changed": ["relative/path/to/file.py"],
  "diff": "apply_op.py JSON result or short description",
  "tests": { "executed": true, "passed": true },
  "warnings": [],
  "confidence": 0.92
}
```

If `confidence` is below 0.70, flag it and ask Claude to review before applying.

---

## FILE_OP format reference

Qwen uses this format when asked to modify or create files.

### REPLACE_EXACT — replace a unique string in a file

```
FILE_OP REPLACE_EXACT
FILE: path/to/file.py
<<<OLD
exact old text (must be unique in the file)
OLD>>>
<<<NEW
replacement text
NEW>>>
END_OP
```

### REPLACE_LINES — replace a line range (1-indexed, inclusive)

```
FILE_OP REPLACE_LINES
FILE: path/to/file.py
FROM: 42
TO: 55
<<<NEW
new content for those lines
NEW>>>
END_OP
```

### INSERT_AFTER — insert code after a unique anchor

```
FILE_OP INSERT_AFTER
FILE: path/to/file.py
<<<AFTER
the exact anchor line or text
AFTER>>>
<<<NEW
code to insert after the anchor
NEW>>>
END_OP
```

### WRITE_FILE — write a new file or complete rewrite

```
FILE_OP WRITE_FILE
FILE: path/to/file.py
<<<CONTENT
full file content here
CONTENT>>>
END_OP
```

---

## Precise task phrasing guide

| Intent | Phrase to use |
|--------|--------------|
| Replace a function body | "Replace the function `foo` with this implementation" |
| Replace lines 200–250 | "Replace lines 200 to 250 in file X with Y" |
| Add a method to a class | "Insert after the line `class Foo:` the following method" |
| Add import at top | "Insert after the last existing import line" |
| Refactor a method | "In file X, where `def bar(` is, replace the whole method with" |
| Create a new file | "Write a new file at path X with content Y" |
| Full file rewrite | "Rewrite the entire file X to do Y (current content below)" |
| Remove dead code | "Replace lines 30 to 45 with nothing (delete them)" |

---

## Full autonomous example

Task: "Add a `__repr__` method to the `QueueItemInfo` dataclass in `monitor/shared.py`"

```powershell
$task = @"
Add a __repr__ method to the QueueItemInfo dataclass.
Return: QueueItem(<first 8 chars of request_id> <method> <path> pri=<client_priority>)

FILE: monitor/shared.py
<<<CONTEXT
@dataclass
class QueueItemInfo:
    request_id: str
    method: str
    path: str
    enqueue_time: float
    body_preview: str
    request_details: str
    request_model: str
    request_prompt: str
    client_key: str
    client_label: str
    client_ip: str
    client_ip_source: str
    client_kind: str
    client_details: str
    client_priority: int
CONTEXT>>>
"@

python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\call_ollama.py" `
  --model qwen2.5-coder:7b `
  --system (Get-Content "$env:USERPROFILE\.claude\skills\ollama-worker\prompts\file_op_system.txt" -Raw) `
  --user $task |
  python "$env:USERPROFILE\.claude\skills\ollama-worker\scripts\apply_op.py"
```

---

## Safety rules

- Never commit, push, or merge
- Never delete files (use REPLACE_LINES with empty NEW to remove lines)
- `apply_op.py` refuses if old text / anchor is ambiguous (multiple matches)
- Run `--dry-run` when confidence < 0.85 or task is large
- If the local model fails or returns no FILE_OP: return `LOCAL_MODEL_UNAVAILABLE`, stop

---

## Roles

| Role | Responsibility |
|------|----------------|
| Claude | Architect, estimates task size, reads context, reviews result |
| Ollama / Qwen | Generates FILE_OP blocks and implementation code |
| apply_op.py | Executes FILE_OP blocks autonomously, reports JSON |
| OpenRouter | Overflow — only when Claude decides to escalate |
