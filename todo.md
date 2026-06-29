# SharedOllama — TODO

## Model: qwen3-coder 8B

- [ ] Decide: pull `qwen3:8b` (official, general) or `freehuntx/qwen3-coder:8b` (community, coder-specific)?
  - `freehuntx/qwen3-coder:8b` is the actual Qwen3-Coder 8B — Alibaba didn't publish it under their own Ollama namespace
  - `qwen3:8b` was 14% downloaded (~710MB of 5.2GB) before we stopped — Ollama may reuse blobs
- [ ] Update `skills/ollama-worker/config.json` with chosen model name
- [ ] Pull chosen model and run `/ollama-worker` end-to-end to verify FILE_OP output quality vs `qwen2.5-coder:7b`

## Skill: FILE_OP operation types

Tested so far:
- [x] WRITE_FILE — works (~12s with qwen2.5-coder:7b)
- [x] REPLACE_EXACT — works end-to-end
- [ ] INSERT_AFTER — not tested solo (only the multi-op edge case was hit)
- [ ] REPLACE_LINES — not tested at all

Test plan for each: write a small test file, ask skill to modify it using that op type, verify apply_op.py applies correctly.

## Admin UI

- [ ] Confirm skill calls appear as a client in admin UI at http://localhost:11444
  - In previous session: skill calls were not visible — fixed stdin hang, but not re-confirmed in UI after fix
  - Check: client name, priority=0, request in history

## Cleanup

- [ ] Consider removing `qwen2.5-coder:1.5b-base` from Ollama (`ollama rm qwen2.5-coder:1.5b-base`) — base model, not useful, takes 1GB
