"""Shared utilities, constants, and state for SharedOllama monitor and proxy."""

import asyncio
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import Request

DEFAULT_CLIENT_PRIORITY = 100
MIN_CLIENT_PRIORITY = 0
MAX_CLIENT_PRIORITY = 1000
DEFAULT_SHARED_PORT = 11435
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_HOST = "127.0.0.1"
DEFAULT_ADMIN_PORT = 11444

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
RUNTIME_CONFIG_PATH = Path(os.getenv("SHAREDOLLAMA_RUNTIME_CONFIG", str(APP_DIR / "runtime_config.json")))
_default_control_script = (
    str(PROJECT_DIR / "scripts" / "ollama_control_windows.ps1")
    if sys.platform == "win32"
    else str(PROJECT_DIR / "scripts" / "wsl_ollama_control.sh")
)
OLLAMA_CONTROL_SCRIPT = Path(os.getenv("SHAREDOLLAMA_OLLAMA_CONTROL_SCRIPT", _default_control_script))


def load_runtime_config() -> dict[str, Any]:
    try:
        if not RUNTIME_CONFIG_PATH.exists():
            return {}
        # PowerShell may write UTF-8 with BOM; utf-8-sig handles both BOM and non-BOM files.
        raw = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_runtime_config(config: dict[str, Any]) -> None:
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_port(raw_value: Any, default_value: int) -> int:
    try:
        parsed = int(str(raw_value).strip())
    except Exception:
        return default_value
    if parsed < 1 or parsed > 65535:
        return default_value
    return parsed


def normalize_host(raw_value: Any) -> str:
    host = str(raw_value or "").strip()
    if not host:
        return DEFAULT_OLLAMA_HOST
    if re.fullmatch(r"[A-Za-z0-9._:-]+", host):
        return host
    return DEFAULT_OLLAMA_HOST


def normalize_backend_url(raw_value: Any, fallback_host: str, fallback_port: int) -> str:
    text = str(raw_value or "").strip().rstrip("/")
    if not text:
        return f"http://{fallback_host}:{fallback_port}"
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"http://{fallback_host}:{fallback_port}"
    return text


async def run_command(command: list[str], timeout_seconds: float = 30.0) -> dict[str, Any]:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return {"ok": False, "exit_code": -1, "stdout": "", "stderr": str(exc)}

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return {"ok": False, "exit_code": -1, "stdout": "", "stderr": "command timeout"}

    out_text = stdout.decode("utf-8", errors="ignore")
    err_text = stderr.decode("utf-8", errors="ignore")
    return {
        "ok": process.returncode == 0,
        "exit_code": process.returncode,
        "stdout": out_text.strip(),
        "stderr": err_text.strip(),
    }


async def run_ollama_control(action: str, host: str, port: int) -> dict[str, Any]:
    if not OLLAMA_CONTROL_SCRIPT.exists():
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Missing control script at {OLLAMA_CONTROL_SCRIPT}",
        }
    if sys.platform == "win32":
        cmd = [
            "powershell.exe", "-NonInteractive", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(OLLAMA_CONTROL_SCRIPT),
            action, "-OllamaHost", host, "-OllamaPort", str(port),
        ]
    else:
        cmd = ["bash", str(OLLAMA_CONTROL_SCRIPT), action, "--host", host, "--port", str(port)]
    return await run_command(cmd, timeout_seconds=40.0)


async def check_backend_version(backend_url: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(f"{backend_url}/api/version")
        body = response.text.strip()
        if response.status_code >= 400:
            return {"ok": False, "status_code": response.status_code, "body": clip_text(body, 300)}
        payload = None
        try:
            payload = response.json()
        except Exception:
            payload = body
        return {"ok": True, "status_code": response.status_code, "body": payload}
    except Exception as exc:
        return {"ok": False, "status_code": None, "body": str(exc)}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip_text(value: Any, max_length: int = 280) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def max_iso(current: str | None, candidate: str | None) -> str | None:
    if not candidate:
        return current
    if not current:
        return candidate
    return candidate if candidate > current else current


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


def parse_priority_value(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return max(MIN_CLIENT_PRIORITY, min(MAX_CLIENT_PRIORITY, parsed))


class MonitorState:
    def __init__(self) -> None:
        runtime_config = load_runtime_config()
        configured_host = normalize_host(runtime_config.get("ollama_host", os.getenv("OLLAMA_HOST_TARGET", DEFAULT_OLLAMA_HOST)))
        configured_port = parse_port(
            runtime_config.get("ollama_port", os.getenv("OLLAMA_PORT_TARGET", str(DEFAULT_OLLAMA_PORT))),
            DEFAULT_OLLAMA_PORT,
        )
        self.shared_port = parse_port(
            runtime_config.get("shared_port", os.getenv("SHAREDOLLAMA_PORT", str(DEFAULT_SHARED_PORT))),
            DEFAULT_SHARED_PORT,
        )
        self.ollama_host = configured_host
        self.ollama_port = configured_port
        self.backend_url = normalize_backend_url(
            runtime_config.get("backend_url", os.getenv("OLLAMA_BACKEND_URL", "")),
            fallback_host=self.ollama_host,
            fallback_port=self.ollama_port,
        )
        self.rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "200"))
        self.workers = int(os.getenv("QUEUE_WORKERS", "2"))
        self.max_log_entries = int(os.getenv("MAX_LOG_ENTRIES", "500"))
        self.max_error_entries = int(os.getenv("MAX_ERROR_ENTRIES", "200"))
        self.max_alert_entries = int(os.getenv("MAX_ALERT_ENTRIES", "200"))
        self.max_history_entries = int(os.getenv("MAX_HISTORY_ENTRIES", "300"))
        self.alert_queue_threshold = int(os.getenv("ALERT_QUEUE_THRESHOLD", "50"))
        self.upstream_timeout = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "300"))
        self.model_pull_timeout = float(os.getenv("MODEL_PULL_TIMEOUT_SECONDS", "900"))
        self.model_pull_retry_count = int(os.getenv("MODEL_PULL_RETRY_COUNT", "1"))
        self.model_refresh_interval_seconds = int(os.getenv("MODEL_REFRESH_INTERVAL_SECONDS", "60"))
        self.monitor_token = os.getenv("MONITOR_TOKEN", "")

        allowlist_raw = os.getenv("MODEL_ALLOWLIST", "*").strip()
        self.model_allowlist = {item.strip() for item in allowlist_raw.split(",") if item.strip()}
        if not self.model_allowlist:
            self.model_allowlist = {"*"}

        self.model_aliases: dict[str, str] = {}
        aliases_raw = os.getenv("MODEL_ALIASES", "").strip()
        if aliases_raw:
            try:
                parsed = json.loads(aliases_raw)
                if isinstance(parsed, dict):
                    self.model_aliases = {str(k): str(v) for k, v in parsed.items()}
            except Exception:
                self.model_aliases = {}

        self.started_at = time.time()
        self.requests_total = 0
        self.requests_in_window = 0
        self.window_start = time.time()
        self.rate_limited_total = 0

        self.processed_total = 0
        self.failed_total = 0

        self.queue: asyncio.PriorityQueue[tuple[int, float, int, dict[str, Any]]] = asyncio.PriorityQueue(
            maxsize=self.max_queue_size
        )
        self.queue_sequence = 0
        self.active = 0
        self.pending: dict[str, QueueItemInfo] = {}
        self.logs: deque[dict[str, Any]] = deque(maxlen=self.max_log_entries)
        self.errors: deque[dict[str, Any]] = deque(maxlen=self.max_error_entries)
        self.alerts: deque[dict[str, Any]] = deque(maxlen=self.max_alert_entries)
        self.history: deque[dict[str, Any]] = deque(maxlen=self.max_history_entries)
        self.client_controls: dict[str, dict[str, Any]] = {}
        self.client_registry: dict[str, dict[str, Any]] = {}
        self.metric_history: deque[dict[str, Any]] = deque(maxlen=240)
        self.models_cache: list[dict[str, Any]] = []
        self.last_models_refresh: str | None = None
        self.model_pull_locks: dict[str, asyncio.Lock] = {}
        self.last_admin_action: str = ""
        self.last_admin_action_at: str | None = None

        self.lock = asyncio.Lock()

    async def runtime_config_snapshot(self) -> dict[str, Any]:
        async with self.lock:
            return {
                "backend_url": self.backend_url,
                "shared_port": self.shared_port,
                "ollama_host": self.ollama_host,
                "ollama_port": self.ollama_port,
                "config_path": str(RUNTIME_CONFIG_PATH),
                "ollama_control_script": str(OLLAMA_CONTROL_SCRIPT),
                "last_admin_action": self.last_admin_action,
                "last_admin_action_at": self.last_admin_action_at,
            }

    async def update_runtime_config(
        self,
        backend_url: str,
        shared_port: int,
        ollama_host: str,
        ollama_port: int,
    ) -> dict[str, Any]:
        normalized_host = normalize_host(ollama_host)
        normalized_port = parse_port(ollama_port, DEFAULT_OLLAMA_PORT)
        normalized_shared_port = parse_port(shared_port, DEFAULT_SHARED_PORT)
        normalized_backend_url = normalize_backend_url(
            backend_url,
            fallback_host=normalized_host,
            fallback_port=normalized_port,
        )

        payload = {
            "backend_url": normalized_backend_url,
            "shared_port": normalized_shared_port,
            "ollama_host": normalized_host,
            "ollama_port": normalized_port,
            "updated_at": now_iso(),
        }
        save_runtime_config(payload)

        async with self.lock:
            self.backend_url = normalized_backend_url
            self.shared_port = normalized_shared_port
            self.ollama_host = normalized_host
            self.ollama_port = normalized_port
            self.last_admin_action = "runtime config updated"
            self.last_admin_action_at = now_iso()

        await self.add_log(
            "INFO",
            "Runtime configuration updated",
            backend_url=normalized_backend_url,
            shared_port=normalized_shared_port,
            ollama_host=normalized_host,
            ollama_port=normalized_port,
        )
        return payload

    async def mark_admin_action(self, action: str, **details: Any) -> None:
        async with self.lock:
            self.last_admin_action = action
            self.last_admin_action_at = now_iso()
        await self.add_log("INFO", f"Admin action: {action}", **details)

    async def add_log(self, level: str, message: str, **extra: Any) -> None:
        entry = {"time": now_iso(), "level": level, "message": message, **extra}
        async with self.lock:
            self.logs.append(entry)
            if level == "ERROR":
                self.errors.append(entry)

    async def add_alert(self, message: str, **extra: Any) -> None:
        entry = {"time": now_iso(), "severity": "warning", "message": message, **extra}
        async with self.lock:
            self.alerts.append(entry)
        await self.add_log("WARN", message, **extra)

    async def check_rate_limit(self) -> bool:
        async with self.lock:
            now = time.time()
            if now - self.window_start >= 60:
                self.window_start = now
                self.requests_in_window = 0
            if self.requests_in_window >= self.rate_limit_per_minute:
                self.rate_limited_total += 1
                return False
            self.requests_in_window += 1
            self.requests_total += 1
            return True

    async def queue_snapshot(self) -> list[dict[str, Any]]:
        async with self.lock:
            items = [asdict(item) for item in self.pending.values()]
        return sorted(items, key=lambda i: (i.get("client_priority", DEFAULT_CLIENT_PRIORITY), i["enqueue_time"]))

    async def add_history_entry(self, entry: dict[str, Any]) -> None:
        async with self.lock:
            self.history.appendleft(entry)

    async def history_snapshot(self) -> list[dict[str, Any]]:
        async with self.lock:
            return list(self.history)

    async def history_item(self, request_id: str) -> dict[str, Any] | None:
        async with self.lock:
            for item in self.history:
                if item.get("request_id") == request_id:
                    return item
        return None

    async def set_client_state(self, client_key: str, state_name: str) -> dict[str, Any]:
        async with self.lock:
            entry = self.client_controls.get(
                client_key,
                {"state": "active", "updated_at": now_iso(), "priority": DEFAULT_CLIENT_PRIORITY},
            )
            entry["state"] = state_name
            entry["updated_at"] = now_iso()
            if "priority" not in entry:
                entry["priority"] = DEFAULT_CLIENT_PRIORITY
            self.client_controls[client_key] = entry
            return {"client_key": client_key, **entry}

    async def register_client(
        self,
        client_key: str,
        client_label: str,
        client_ip: str,
        client_ip_source: str,
        client_kind: str,
        client_details: str,
    ) -> None:
        async with self.lock:
            existing = self.client_registry.get(client_key, {})
            self.client_registry[client_key] = {
                "client_key": client_key,
                "client_label": client_label or existing.get("client_label", client_key),
                "client_ip": client_ip or existing.get("client_ip", ""),
                "client_ip_source": client_ip_source or existing.get("client_ip_source", "unknown"),
                "client_kind": client_kind or existing.get("client_kind", "unknown"),
                "client_details": client_details or existing.get("client_details", ""),
                "last_seen": now_iso(),
            }

            control_entry = self.client_controls.get(client_key)
            if control_entry is None:
                self.client_controls[client_key] = {
                    "state": "active",
                    "updated_at": now_iso(),
                    "priority": DEFAULT_CLIENT_PRIORITY,
                }

    async def set_client_priority(self, client_key: str, priority: int, source: str = "manual") -> dict[str, Any]:
        normalized_priority = max(MIN_CLIENT_PRIORITY, min(MAX_CLIENT_PRIORITY, int(priority)))
        async with self.lock:
            entry = self.client_controls.get(
                client_key,
                {"state": "active", "updated_at": now_iso(), "priority": DEFAULT_CLIENT_PRIORITY},
            )
            entry["priority"] = normalized_priority
            entry["updated_at"] = now_iso()
            entry["priority_source"] = source
            self.client_controls[client_key] = entry
            return {"client_key": client_key, **entry}

    async def get_client_state(self, client_key: str) -> str:
        async with self.lock:
            entry = self.client_controls.get(client_key)
            if not entry:
                return "active"
            return str(entry.get("state", "active"))

    async def get_client_priority(self, client_key: str) -> int:
        async with self.lock:
            entry = self.client_controls.get(client_key)
            if not entry:
                return DEFAULT_CLIENT_PRIORITY
            parsed_priority = parse_priority_value(entry.get("priority"))
            if parsed_priority is None:
                return DEFAULT_CLIENT_PRIORITY
            return parsed_priority

    async def record_metric_snapshot(self) -> None:
        async with self.lock:
            self.metric_history.append(
                {
                    "time": now_iso(),
                    "ts": time.time(),
                    "requests_total": self.requests_total,
                    "queued_count": len(self.pending),
                    "processed_total": self.processed_total,
                    "failed_total": self.failed_total,
                }
            )

    async def client_snapshot(self) -> list[dict[str, Any]]:
        async with self.lock:
            queue_items = [asdict(item) for item in self.pending.values()]
            history_items = list(self.history)
            control_map = dict(self.client_controls)
            registry_map = dict(self.client_registry)

        summary: dict[str, dict[str, Any]] = {}
        for client_key, item in registry_map.items():
            summary[client_key] = {
                "client_key": client_key,
                "client_label": item.get("client_label", client_key),
                "client_ip": item.get("client_ip", ""),
                "client_ip_source": item.get("client_ip_source", "unknown"),
                "client_kind": item.get("client_kind", "unknown"),
                "client_details": item.get("client_details", ""),
                "client_priority": int(control_map.get(client_key, {}).get("priority", DEFAULT_CLIENT_PRIORITY)),
                "queued_count": 0,
                "completed_count": 0,
                "failed_count": 0,
                "last_seen": item.get("last_seen"),
                "state": control_map.get(client_key, {}).get("state", "active"),
                "priority": int(control_map.get(client_key, {}).get("priority", DEFAULT_CLIENT_PRIORITY)),
                "updated_at": control_map.get(client_key, {}).get("updated_at"),
            }

        for item in queue_items:
            client_key = str(item.get("client_key", "unknown"))
            entry = summary.setdefault(
                client_key,
                {
                    "client_key": client_key,
                    "client_label": item.get("client_label", client_key),
                    "client_ip": item.get("client_ip", ""),
                    "client_ip_source": item.get("client_ip_source", "unknown"),
                    "client_kind": item.get("client_kind", "unknown"),
                    "client_details": item.get("client_details", ""),
                    "client_priority": int(item.get("client_priority", DEFAULT_CLIENT_PRIORITY)),
                    "queued_count": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "last_seen": None,
                    "state": control_map.get(client_key, {}).get("state", "active"),
                    "priority": int(control_map.get(client_key, {}).get("priority", DEFAULT_CLIENT_PRIORITY)),
                    "updated_at": control_map.get(client_key, {}).get("updated_at"),
                },
            )
            entry["queued_count"] += 1
            entry["last_seen"] = max_iso(entry["last_seen"], now_iso())
            entry["priority"] = int(
                control_map.get(client_key, {}).get("priority", entry.get("client_priority", DEFAULT_CLIENT_PRIORITY))
            )

        for item in history_items:
            client_key = str(item.get("client_key", "unknown"))
            entry = summary.setdefault(
                client_key,
                {
                    "client_key": client_key,
                    "client_label": item.get("client_label", client_key),
                    "client_ip": item.get("client_ip", ""),
                    "client_ip_source": item.get("client_ip_source", "unknown"),
                    "client_kind": item.get("client_kind", "unknown"),
                    "client_details": item.get("client_details", ""),
                    "client_priority": int(item.get("client_priority", DEFAULT_CLIENT_PRIORITY)),
                    "queued_count": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "last_seen": None,
                    "state": control_map.get(client_key, {}).get("state", "active"),
                    "priority": int(control_map.get(client_key, {}).get("priority", DEFAULT_CLIENT_PRIORITY)),
                    "updated_at": control_map.get(client_key, {}).get("updated_at"),
                },
            )
            if item.get("error") or (item.get("status_code") is not None and int(item["status_code"]) >= 400):
                entry["failed_count"] += 1
            else:
                entry["completed_count"] += 1
            completed_at = str(item.get("completed_at") or "")
            if completed_at:
                entry["last_seen"] = max_iso(entry["last_seen"], completed_at)
            entry["priority"] = int(
                control_map.get(client_key, {}).get("priority", entry.get("client_priority", DEFAULT_CLIENT_PRIORITY))
            )

        return sorted(
            summary.values(),
            key=lambda item: (item["priority"], -item["queued_count"], -item["completed_count"]),
        )

    async def refresh_models(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(f"{self.backend_url}/api/tags")
                r.raise_for_status()
                payload = r.json()
            models = payload.get("models", [])
            async with self.lock:
                self.models_cache = models if isinstance(models, list) else []
                self.last_models_refresh = now_iso()
            await self.add_log("INFO", "Models refreshed", count=len(self.models_cache))
        except Exception as exc:
            await self.add_log("ERROR", "Failed refreshing models", error=str(exc))

    def resolve_model_name(self, requested_model: str) -> str:
        model = requested_model.strip()
        if not model:
            return model
        return self.model_aliases.get(model, model)

    def model_allowed(self, model_name: str) -> bool:
        if "*" in self.model_allowlist:
            return True

        for item in self.model_allowlist:
            if item.endswith("*"):
                if model_name.startswith(item[:-1]):
                    return True
            elif model_name == item:
                return True
        return False

    async def get_cached_model_names(self) -> set[str]:
        async with self.lock:
            names = {
                str(m.get("name", "")).strip()
                for m in self.models_cache
                if isinstance(m, dict) and str(m.get("name", "")).strip()
            }
        return names

    async def ensure_model_available(self, model_name: str, request_id: str) -> bool:
        if not model_name:
            return True

        cached_names = await self.get_cached_model_names()
        if model_name in cached_names:
            return True

        async with self.lock:
            pull_lock = self.model_pull_locks.get(model_name)
            if pull_lock is None:
                pull_lock = asyncio.Lock()
                self.model_pull_locks[model_name] = pull_lock

        async with pull_lock:
            cached_names = await self.get_cached_model_names()
            if model_name in cached_names:
                return True

            await self.add_log("INFO", "Model missing, starting pull", request_id=request_id, model=model_name)
            try:
                async with httpx.AsyncClient(timeout=self.model_pull_timeout) as client:
                    pull_response = await client.post(
                        f"{self.backend_url}/api/pull",
                        json={"name": model_name, "stream": False},
                    )
            except Exception as exc:
                await self.add_log(
                    "ERROR",
                    "Model pull failed with exception",
                    request_id=request_id,
                    model=model_name,
                    error=str(exc),
                )
                return False

            if pull_response.status_code >= 400:
                await self.add_log(
                    "ERROR",
                    "Model pull returned error",
                    request_id=request_id,
                    model=model_name,
                    status_code=pull_response.status_code,
                )
                return False

            await self.add_log("INFO", "Model pulled successfully", request_id=request_id, model=model_name)
            await self.refresh_models()
            return True


async def ensure_monitor_auth(request: Request | None, monitor_token: str = "") -> Any | None:
    if not monitor_token:
        return None
    if request is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    token = request.headers.get("x-monitor-token") or request.query_params.get("token", "")
    if token != monitor_token:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None
