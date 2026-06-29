"""
Call a local Ollama model via SharedOllama (http://localhost:11434).

Usage:
  python call_ollama.py --model qwen2.5:7b --system "You are a coding assistant." --user "Write a Python hello world"
  echo "def foo(): pass" | python call_ollama.py --model qwen2.5:7b --user "Add a docstring to this function"

Flags:
  --model     Ollama model name (required)
  --system    System prompt (optional, defaults to coding assistant)
  --user      User message (required unless piped via stdin)
  --url       Ollama base URL (default: http://localhost:11434)
  --timeout   Request timeout in seconds (default: 120)
  --json      Print full JSON response instead of just the content
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


DEFAULT_SYSTEM = (
    "You are a focused coding assistant. "
    "Produce correct, minimal, well-named code. "
    "Return only the requested output — no explanations unless asked."
)


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
            "x-client-priority": "0",   # highest priority in SharedOllama queue
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
    parser.add_argument("--model",   required=True, help="Ollama model name")
    parser.add_argument("--system",  default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--user",    default=None, help="User message (or pipe via stdin)")
    parser.add_argument("--url",     default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    parser.add_argument("--json",    action="store_true", help="Print full JSON response")
    args = parser.parse_args()

    # Accept user message from stdin if not provided as argument
    if args.user is None:
        if sys.stdin.isatty():
            parser.error("--user is required when stdin is not piped")
        stdin_content = sys.stdin.read().strip()
        user_message = stdin_content
    else:
        user_message = args.user
        # Prepend stdin content if also piped
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()
            if stdin_content:
                user_message = f"{stdin_content}\n\n{user_message}"

    result = call_ollama(
        model=args.model,
        system=args.system,
        user=user_message,
        base_url=args.url,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        content = result.get("message", {}).get("content", "")
        print(content)


if __name__ == "__main__":
    main()
