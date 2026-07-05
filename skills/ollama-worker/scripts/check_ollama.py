#!/usr/bin/env python3
"""Quick Ollama health check for ollama-worker."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().parent / "call_ollama.py"

    list_result = subprocess.run(
        [sys.executable, str(script), "--list"],
        capture_output=True,
        text=True,
    )
    if list_result.returncode != 0:
        print(list_result.stderr, file=sys.stderr)
        print("FAIL: could not reach Ollama or config is invalid", file=sys.stderr)
        return 1

    installed = []
    for line in list_result.stdout.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "yes":
            installed.append(parts[0])

    print(list_result.stdout, end="")
    if not installed:
        print("FAIL: Ollama is reachable but no preferred model is installed", file=sys.stderr)
        return 1

    print("preferred model available")

    live_result = subprocess.run(
        [sys.executable, str(script), "--user", "Reply with exactly: OK"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if live_result.returncode != 0:
        print(live_result.stderr, file=sys.stderr)
        print("FAIL: live Ollama call failed", file=sys.stderr)
        return live_result.returncode or 1

    content = live_result.stdout.strip()
    print(f"live response: {content!r}")
    print(json.dumps({"ok": True, "response": content}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
