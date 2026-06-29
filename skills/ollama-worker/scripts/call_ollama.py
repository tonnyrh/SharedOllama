"""
Call a local Ollama model via SharedOllama (http://localhost:11434).

Model selection: if --model is omitted, the script reads config.json from the
skill directory and picks the first model in the preferred list that is actually
installed in Ollama. Edit config.json to match your hardware.

Usage:
  python call_ollama.py --user "Write a Python hello world"
  python call_ollama.py --model qwen2.5-coder:7b --user "Add a docstring to foo"
  echo "def foo(): pass" | python call_ollama.py --user "Add a docstring"

Flags:
  --model     Ollama model name (optional — auto-selected from config.json if omitted)
  --system    System prompt (optional, defaults to coding assistant)
  --user      User message (required unless piped via stdin)
  --url       Ollama base URL (overrides config.json if provided)
  --timeout   Request timeout in seconds (default: 120)
  --json      Print full JSON response instead of just the content
  --list      List available models from config and exit
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path


DEFAULT_SYSTEM = (
    "You are a focused coding assistant. "
    "Produce correct, minimal, well-named code. "
    "Return only the requested output — no explanations unless asked."
)

SKILL_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = SKILL_DIR / "config.json"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _installed_models(base_url: str) -> set[str]:
    """Return the set of model names currently installed in Ollama."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


def _select_model(config: dict, base_url: str) -> str | None:
    """Pick the first preferred model that is installed in Ollama."""
    preferred = [m["name"] for m in config.get("models", [])]
    if not preferred:
        return None
    installed = _installed_models(base_url)
    for name in preferred:
        if name in installed:
            return name
    return None


def call_ollama(model: str, system: str, user: str, base_url: str, timeout: int) -> dict:
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-client-priority": "0",
            "x-client-name": "ollama-worker-skill",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"LOCAL_MODEL_UNAVAILABLE: {exc}", file=sys.stderr)
        sys.exit(2)
    except TimeoutError:
        print("LOCAL_MODEL_UNAVAILABLE: request timed out", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a local Ollama model via SharedOllama.")
    parser.add_argument("--model",   default=None, help="Model name (auto-selected from config.json if omitted)")
    parser.add_argument("--system",  default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--user",    default=None, help="User message (or pipe via stdin)")
    parser.add_argument("--url",     default=None, help="Ollama base URL (overrides config.json)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    parser.add_argument("--json",    action="store_true", help="Print full JSON response")
    parser.add_argument("--list",    action="store_true", help="List preferred models from config and exit")
    args = parser.parse_args()

    config = _load_config()
    base_url = args.url or config.get("ollama_url", "http://localhost:11434")

    if args.list:
        installed = _installed_models(base_url)
        preferred = config.get("models", [])
        if not preferred:
            print("No models configured in config.json", file=sys.stderr)
            sys.exit(1)
        print(f"{'MODEL':<35} {'INSTALLED':<10} CONTEXT    NOTE")
        print("-" * 80)
        for m in preferred:
            name = m["name"]
            ctx = f"{m.get('context_tokens', '?'):,}"
            flag = "yes" if name in installed else "no"
            note = m.get("note", "")
            print(f"{name:<35} {flag:<10} {ctx:<10} {note}")
        sys.exit(0)

    # Resolve model — explicit flag wins, then auto-select from config
    model = args.model
    if not model:
        model = _select_model(config, base_url)
        if not model:
            print(
                "LOCAL_MODEL_UNAVAILABLE: no model specified and none of the preferred "
                "models from config.json are installed. Run --list to see options.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"[ollama-worker] auto-selected model: {model}", file=sys.stderr)

    # Resolve user message
    if args.user is None:
        if sys.stdin.isatty():
            parser.error("--user is required when stdin is not piped")
        user_message = sys.stdin.read().strip()
    else:
        user_message = args.user

    result = call_ollama(
        model=model,
        system=args.system,
        user=user_message,
        base_url=base_url,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        content = result.get("message", {}).get("content", "")
        print(content)


if __name__ == "__main__":
    main()
