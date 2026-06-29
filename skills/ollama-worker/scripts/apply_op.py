"""
Parse FILE_OP blocks from Qwen output and apply them to the local filesystem.

Usage:
    python apply_op.py [response_file]           # read from file
    python apply_op.py --dry-run [response_file] # preview without writing
    cat qwen_output.txt | python apply_op.py     # read from stdin

Exits 0 on success, 1 on parse error, 2 on apply error.
Always prints a JSON result to stdout.

--- FILE_OP FORMAT ---

Four operation types are supported. Each block starts with FILE_OP <TYPE>
and ends with END_OP. Delimiters are <<<TAG and TAG>>>.

REPLACE_EXACT — replace an exact string (must be unique in the file):

    FILE_OP REPLACE_EXACT
    FILE: path/to/file.py
    <<<OLD
    exact old text here
    can be multiple lines
    OLD>>>
    <<<NEW
    replacement text
    NEW>>>
    END_OP

REPLACE_LINES — replace a line range (1-indexed, inclusive):

    FILE_OP REPLACE_LINES
    FILE: path/to/file.py
    FROM: 42
    TO: 55
    <<<NEW
    new content for those lines
    NEW>>>
    END_OP

INSERT_AFTER — insert code after a unique anchor string:

    FILE_OP INSERT_AFTER
    FILE: path/to/file.py
    <<<AFTER
    the exact anchor line or text
    AFTER>>>
    <<<NEW
    code to insert after the anchor
    NEW>>>
    END_OP

WRITE_FILE — write a new file (or overwrite an existing one):

    FILE_OP WRITE_FILE
    FILE: path/to/new_file.py
    <<<CONTENT
    full file content here
    CONTENT>>>
    END_OP
"""

import json
import re
import sys
from pathlib import Path


def _extract_block(body: str, tag: str) -> str | None:
    m = re.search(rf"<<<{tag}\n(.*?)\n{tag}>>>", body, re.DOTALL)
    return m.group(1) if m else None


def _parse_ops(text: str) -> list[dict]:
    ops = []
    for match in re.finditer(r"FILE_OP\s+(\w+)\s*\n(.*?)END_OP", text, re.DOTALL):
        op_type = match.group(1).strip()
        body = match.group(2)
        op: dict = {"type": op_type}

        file_m = re.search(r"^FILE:\s*(.+)$", body, re.MULTILINE)
        if file_m:
            op["file"] = file_m.group(1).strip()

        from_m = re.search(r"^FROM:\s*(\d+)$", body, re.MULTILINE)
        to_m = re.search(r"^TO:\s*(\d+)$", body, re.MULTILINE)
        if from_m:
            op["from_line"] = int(from_m.group(1))
        if to_m:
            op["to_line"] = int(to_m.group(1))

        for tag in ("OLD", "NEW", "AFTER", "CONTENT"):
            val = _extract_block(body, tag)
            if val is not None:
                op[tag.lower()] = val

        ops.append(op)
    return ops


def _replace_exact(op: dict, dry_run: bool) -> dict:
    path = Path(op["file"])
    if not path.exists():
        return {"ok": False, "error": f"File not found: {op['file']}"}

    text = path.read_text(encoding="utf-8")
    old = op.get("old", "")
    new = op.get("new", "")

    count = text.count(old)
    if count == 0:
        return {"ok": False, "error": f"Old text not found in {op['file']}"}
    if count > 1:
        return {"ok": False, "error": f"Old text is ambiguous ({count} matches) — make it more specific"}

    if not dry_run:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return {"ok": True, "file": op["file"], "op": "REPLACE_EXACT", "dry_run": dry_run}


def _replace_lines(op: dict, dry_run: bool) -> dict:
    path = Path(op["file"])
    if not path.exists():
        return {"ok": False, "error": f"File not found: {op['file']}"}

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    from_idx = op.get("from_line", 1) - 1
    to_idx = op.get("to_line", len(lines))

    if not (0 <= from_idx < len(lines)):
        return {"ok": False, "error": f"from_line {op.get('from_line')} out of range (file has {len(lines)} lines)"}

    new_content = op.get("new", "")
    new_lines = new_content.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    result = lines[:from_idx] + new_lines + lines[to_idx:]
    if not dry_run:
        path.write_text("".join(result), encoding="utf-8")
    return {"ok": True, "file": op["file"], "op": "REPLACE_LINES",
            "from": op.get("from_line"), "to": op.get("to_line"), "dry_run": dry_run}


def _insert_after(op: dict, dry_run: bool) -> dict:
    path = Path(op["file"])
    if not path.exists():
        return {"ok": False, "error": f"File not found: {op['file']}"}

    text = path.read_text(encoding="utf-8")
    anchor = op.get("after", "")
    new = op.get("new", "")

    count = text.count(anchor)
    if count == 0:
        return {"ok": False, "error": f"Anchor not found in {op['file']}"}
    if count > 1:
        return {"ok": False, "error": f"Anchor is ambiguous ({count} matches) — make it more specific"}

    insert_pos = text.index(anchor) + len(anchor)
    new_text = text[:insert_pos] + "\n" + new + text[insert_pos:]
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return {"ok": True, "file": op["file"], "op": "INSERT_AFTER", "dry_run": dry_run}


def _write_file(op: dict, dry_run: bool) -> dict:
    path = Path(op["file"])
    content = op.get("content", "")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return {"ok": True, "file": op["file"], "op": "WRITE_FILE", "dry_run": dry_run}


_HANDLERS = {
    "REPLACE_EXACT": _replace_exact,
    "REPLACE_LINES": _replace_lines,
    "INSERT_AFTER": _insert_after,
    "WRITE_FILE": _write_file,
}


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    text = Path(args[0]).read_text(encoding="utf-8") if args else sys.stdin.read()

    ops = _parse_ops(text)
    if not ops:
        print(json.dumps({"ok": False, "error": "No FILE_OP blocks found", "ops_applied": []}))
        sys.exit(1)

    results = []
    all_ok = True
    for op in ops:
        handler = _HANDLERS.get(op.get("type", ""))
        if handler is None:
            result = {"ok": False, "error": f"Unknown op type: {op.get('type')}"}
        else:
            result = handler(op, dry_run)
        results.append(result)
        if not result.get("ok"):
            all_ok = False
            break  # stop on first failure to avoid cascading errors

    print(json.dumps({"ok": all_ok, "ops_applied": results}, indent=2))
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
