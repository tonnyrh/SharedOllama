---
name: ollama-worker
description: Delegate small, bounded coding and editing tasks in SharedOllama to the local Ollama/Qwen worker. Use for one-file changes, one function or method, targeted refactors, focused tests, small scripts, regex, documentation edits, and FILE_OP-based patch generation. Read the canonical skill in ../../../skills/ollama-worker before invoking scripts.
---

# Cursor Wrapper

This wrapper exists so Cursor discovers the project skill under `.cursor/skills`.

Canonical skill:
- [../../../skills/ollama-worker/SKILL.md](../../../skills/ollama-worker/SKILL.md)

Executable resources:
- [../../../skills/ollama-worker/scripts/call_ollama.py](../../../skills/ollama-worker/scripts/call_ollama.py)
- [../../../skills/ollama-worker/scripts/apply_op.py](../../../skills/ollama-worker/scripts/apply_op.py)
- [../../../skills/ollama-worker/prompts/file_op_system.txt](../../../skills/ollama-worker/prompts/file_op_system.txt)

Before using this wrapper, read the canonical skill and follow its task-sizing, FILE_OP, dry-run, and root-safety rules.

