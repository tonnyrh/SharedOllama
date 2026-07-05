---
name: ollama-worker
description: Delegate small, bounded coding and editing work to a local Ollama/Qwen worker before using cloud models. Use for one-file changes, one function or method, targeted refactors, focused tests, small scripts, regex, documentation edits, and FILE_OP-based patch generation where the primary agent remains architect and reviewer. Do not use for architecture, multi-subsystem changes, ambiguous requirements, security-sensitive analysis, or tasks that exceed local context; escalate those to OpenRouter only after the primary agent decides the local worker is unsuitable.
---

# Local Ollama Worker

Delegate small, well-defined coding tasks to the local Ollama model via SharedOllama. The primary agent remains architect, context selector, reviewer, and test runner. The local model handles implementation drafts and structured FILE_OP edits only.

Use this skill as the default offload path for simple implementation work. Use OpenRouter/GLM only when the work is too broad, too ambiguous, too risky, or too context-heavy for the local worker.

---

## Task sizing — the primary agent decides before delegating

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

**Qwen can produce whole files** - use `WRITE_FILE` for new files or complete rewrites.
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
- Tasks where the needed context cannot fit with margin

If the task does not qualify, handle it directly or use `openrouter-heavy-task-gate` for a larger second pass. Do not stop for user confirmation unless the route requires missing credentials, sensitive context, destructive operations, or unclear requirements.

---

## Workflow

### Step 1 - Set the skill path and check model availability

```powershell
$skill = "C:\vscode\SharedOllama\skills\ollama-worker"
python "$skill\scripts\call_ollama.py" --list
```

`call_ollama.py` auto-selects the first installed model from `config.json` when `--model` is omitted. Override `--model` only when task size or quality requires it.

Preferred model order:

1. `qwen3-coder` or `qwen3` coder-capable variants - largest context, best quality
2. `qwen2.5-coder:7b` - good quality, 32K context
3. `qwen2.5-coder:1.5b` - lightweight, small tasks only
4. `qwen2.5` (any variant) — fallback
5. Any available model

Adjust model choice based on task size (see Task sizing above).
If Ollama is unreachable or no model is available, return `LOCAL_MODEL_UNAVAILABLE` and stop.
The primary agent may then continue directly with normal local work or `openrouter-heavy-task-gate` when the task is heavy enough.

### Step 2 - Gather minimal context

Use Read, Grep, or Glob tools. Read the minimum needed:

- For targeted edits: read only the relevant lines/function
- For whole-file rewrites: read the full file (only if it fits the model)
- Use Grep to locate relevant functions or symbols
- Never read unrelated files
- Include exact file paths relative to the repository root whenever possible
- State the desired operation explicitly: replace exact block, replace lines, insert after anchor, or write file

### Step 3 - Call Ollama

**For pure generation tasks** (new function, new file, explain code):

```powershell
python "$skill\scripts\call_ollama.py" `
  --system "You are a focused coding assistant. Return only the requested code." `
  --user "<task + minimal context>"
```

**For file modification tasks** (edit, replace, inject, whole-file rewrite) — use the
file-op system prompt so Qwen returns a structured FILE_OP block that `apply_op.py`
can execute autonomously:

```powershell
python "$skill\scripts\call_ollama.py" `
  --system (Get-Content -LiteralPath "$skill\prompts\file_op_system.txt" -Raw) `
  --user "<task>\n\nFILE: <path>\n<relevant content>"
```

Always use:
- `"stream": false` - routes through SharedOllama priority queue
- `"x-client-priority": "0"` - highest priority; code assistance served first

### Step 4 - Apply file operations

```powershell
$repo = (Resolve-Path -LiteralPath .).Path
$opFile = Join-Path $env:TEMP "ollama-worker-file-op.txt"

python "$skill\scripts\call_ollama.py" ... | Set-Content -LiteralPath $opFile -Encoding utf8
python "$skill\scripts\apply_op.py" --root $repo --dry-run $opFile
python "$skill\scripts\apply_op.py" --root $repo $opFile
```

`apply_op.py` exits 0 on success, 1 on parse error, 2 on apply error, and always prints
a JSON result. Stop and review locally on any non-zero exit. Do not ask Ollama to repair blindly until the primary agent has inspected the failure.

### Step 5 - Review and test

Review the diff after applying operations. Keep, adjust, or discard by normal local editing rules; the local worker does not own final correctness.

If the project has a known test command, run it and note the result.

### Step 6 - Return structured result

Always end with:

```json
{
  "summary": "One sentence describing what was done",
  "files_changed": ["relative/path/to/file.py"],
  "apply_result": "apply_op.py JSON result or short description",
  "tests": { "executed": true, "passed": true },
  "warnings": [],
  "confidence": 0.92
}
```

If `confidence` is below 0.70, flag it and review before applying.

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
$skill = "C:\vscode\SharedOllama\skills\ollama-worker"
$repo = (Resolve-Path -LiteralPath .).Path
$opFile = Join-Path $env:TEMP "ollama-worker-file-op.txt"

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

python "$skill\scripts\call_ollama.py" `
  --system (Get-Content -LiteralPath "$skill\prompts\file_op_system.txt" -Raw) `
  --user $task | Set-Content -LiteralPath $opFile -Encoding utf8

python "$skill\scripts\apply_op.py" --root $repo --dry-run $opFile
python "$skill\scripts\apply_op.py" --root $repo $opFile
```

---

## Safety rules

- Never commit, push, or merge
- Never delete files (use REPLACE_LINES with empty NEW to remove lines)
- Always pass `--root` to `apply_op.py` so FILE_OP paths cannot escape the intended workspace
- `apply_op.py` refuses if old text / anchor is ambiguous (multiple matches)
- Run `--dry-run` when confidence < 0.85 or task is large
- If the local model fails or returns no FILE_OP: return `LOCAL_MODEL_UNAVAILABLE`, stop

---

## Roles

| Role | Responsibility |
|------|----------------|
| Primary agent | Architect, estimates task size, reads context, reviews result |
| Ollama / Qwen | Generates FILE_OP blocks and implementation code |
| apply_op.py | Executes FILE_OP blocks autonomously, reports JSON |
| OpenRouter | Overflow for larger, riskier, or long-context work after the agent decides to escalate |

## Health check

```powershell
$skill = "C:\vscode\SharedOllama\skills\ollama-worker"
python "$skill\scripts\call_ollama.py" --list
python "$skill\scripts\check_ollama.py"
```
