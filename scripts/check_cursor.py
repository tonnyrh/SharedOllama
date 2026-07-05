#!/usr/bin/env python3
"""Verify SharedOllama Cursor skill files and optional local Ollama health."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SharedOllama Cursor skill setup.")
    parser.add_argument("--live-ollama", action="store_true", help="Run a live Ollama call.")
    return parser.parse_args()


def syntax_check(paths: list[Path]) -> None:
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"))
        print(f"syntax OK: {path}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    skill = repo_root / "skills" / "ollama-worker"

    required = [
        skill / "SKILL.md",
        repo_root / ".cursor" / "skills" / "ollama-worker" / "SKILL.md",
        repo_root / ".cursor" / "rules" / "agent-routing.mdc",
        repo_root / "AGENTS.md",
        skill / "scripts" / "call_ollama.py",
        skill / "scripts" / "apply_op.py",
        skill / "scripts" / "check_ollama.py",
        skill / "prompts" / "file_op_system.txt",
        skill / "config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        for path in missing:
            print(f"FAIL: missing {path}", file=sys.stderr)
        return 1

    syntax_check([
        skill / "scripts" / "call_ollama.py",
        skill / "scripts" / "apply_op.py",
        skill / "scripts" / "check_ollama.py",
    ])

    list_result = subprocess.run(
        [sys.executable, str(skill / "scripts" / "call_ollama.py"), "--list"],
        capture_output=True,
        text=True,
    )
    if list_result.returncode != 0:
        print("WARN: Ollama is not reachable from this environment", file=sys.stderr)
        print(list_result.stderr, file=sys.stderr)
    else:
        print(list_result.stdout, end="")
        print("ollama list OK")

    if args.live_ollama:
        subprocess.run([sys.executable, str(skill / "scripts" / "check_ollama.py")], check=True)

    print("Cursor skill layout OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
