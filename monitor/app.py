import asyncio
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from ipaddress import ip_address
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

MAX_BODY_PREVIEW_LENGTH = 200
MAX_HISTORY_PREVIEW_LENGTH = 280
MAX_DETAIL_TEXT_LENGTH = 12000
BINARY_CONTENT_PREFIXES = ("image/", "audio/", "video/")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class MonitorState:
    def __init__(self) -> None:
        self.backend_url = os.getenv("OLLAMA_BACKEND_URL", "http://ollama:11434").rstrip("/")
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

        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.max_queue_size)
        self.active = 0
        self.pending: dict[str, QueueItemInfo] = {}
        self.logs: deque[dict[str, Any]] = deque(maxlen=self.max_log_entries)
        self.errors: deque[dict[str, Any]] = deque(maxlen=self.max_error_entries)
        self.alerts: deque[dict[str, Any]] = deque(maxlen=self.max_alert_entries)
        self.history: deque[dict[str, Any]] = deque(maxlen=self.max_history_entries)
        self.client_controls: dict[str, dict[str, Any]] = {}
        self.metric_history: deque[dict[str, Any]] = deque(maxlen=240)
        self.models_cache: list[dict[str, Any]] = []
        self.last_models_refresh: str | None = None
        self.model_pull_locks: dict[str, asyncio.Lock] = {}

        self.lock = asyncio.Lock()

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
        return sorted(items, key=lambda i: i["enqueue_time"])

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
            entry = self.client_controls.get(client_key, {"state": "active", "updated_at": now_iso()})
            entry["state"] = state_name
            entry["updated_at"] = now_iso()
            self.client_controls[client_key] = entry
            return {"client_key": client_key, **entry}

    async def get_client_state(self, client_key: str) -> str:
        async with self.lock:
            entry = self.client_controls.get(client_key)
            if not entry:
                return "active"
            return str(entry.get("state", "active"))

    async def cancel_pending_for_client(self, client_key: str, reason: str) -> int:
        drained: list[dict[str, Any]] = []
        removed = 0

        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if item.get("client_key") == client_key:
                removed += 1
                future = item.get("future")
                if future is not None and not future.done():
                    future.set_exception(RuntimeError(reason))
                request_id = str(item.get("request_id", ""))
                async with self.lock:
                    self.pending.pop(request_id, None)
                self.queue.task_done()
            else:
                drained.append(item)
                self.queue.task_done()

        for item in drained:
            await self.queue.put(item)

        return removed

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

        summary: dict[str, dict[str, Any]] = {}
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
                    "queued_count": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "last_seen": None,
                    "state": control_map.get(client_key, {}).get("state", "active"),
                    "updated_at": control_map.get(client_key, {}).get("updated_at"),
                },
            )
            entry["queued_count"] += 1
            entry["last_seen"] = max_iso(entry["last_seen"], now_iso())

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
                    "queued_count": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "last_seen": None,
                    "state": control_map.get(client_key, {}).get("state", "active"),
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

        return sorted(summary.values(), key=lambda item: (item["queued_count"], item["completed_count"]), reverse=True)

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
                    body_preview=clip_text(pull_response.text, 300),
                )
                return False

            await self.refresh_models()
            cached_names = await self.get_cached_model_names()
            is_available = model_name in cached_names
            if is_available:
                await self.add_log("INFO", "Model pull completed", request_id=request_id, model=model_name)
            else:
                await self.add_log("ERROR", "Model pull completed but model still unavailable", request_id=request_id, model=model_name)
            return is_available

    async def process_queue(self) -> None:
        while True:
            item = await self.queue.get()
            request_id = item["request_id"]
            future = item["future"]
            async with self.lock:
                self.active += 1

            started = time.time()
            try:
                async with httpx.AsyncClient(timeout=self.upstream_timeout) as client:
                    resp = await client.request(
                        method=item["method"],
                        url=f'{self.backend_url}{item["path"]}',
                        content=item["body"],
                        headers=item["headers"],
                    )
                result = {
                    "status_code": resp.status_code,
                    "content": resp.content,
                    "headers": dict(resp.headers),
                }
                if not future.done():
                    future.set_result(result)
                async with self.lock:
                    self.processed_total += 1

                req_content_type = item["headers"].get("content-type", "")
                resp_content_type = resp.headers.get("content-type", "")
                req_parsed = safe_json_parse(item["body"], req_content_type)
                resp_parsed = safe_json_parse(resp.content, resp_content_type)
                req_fields = extract_request_fields(req_parsed)
                resp_fields = extract_response_fields(resp_parsed)

                request_preview = make_body_preview(item["body"], req_content_type)
                response_preview = make_body_preview(resp.content, resp_content_type)
                question_preview = get_user_question(req_parsed) or clip_text(request_preview)
                answer_preview = get_model_answer(resp_parsed) or clip_text(response_preview)

                await self.add_history_entry(
                    {
                        "request_id": request_id,
                        "method": item["method"],
                        "path": item["path"],
                        "client_key": item.get("client_key", "unknown"),
                        "client_label": item.get("client_label", "unknown"),
                        "client_ip": item.get("client_ip", "unknown"),
                        "client_ip_source": item.get("client_ip_source", "unknown"),
                        "client_kind": item.get("client_kind", "unknown"),
                        "client_details": item.get("client_details", ""),
                        "status_code": resp.status_code,
                        "duration_ms": round((time.time() - started) * 1000, 2),
                        "enqueue_time": item.get("enqueue_time"),
                        "completed_at": now_iso(),
                        "question_preview": question_preview,
                        "answer_preview": answer_preview,
                        "request_preview": request_preview,
                        "response_preview": response_preview,
                        "request_details": detail_text(item["body"], req_content_type),
                        "response_details": detail_text(resp.content, resp_content_type, omit_context=True),
                        "request_model": req_fields["model"],
                        "request_prompt": req_fields["prompt"],
                        "response_model": resp_fields["model"],
                        "response_text": resp_fields["response"],
                        "response_created_at": resp_fields["created_at"],
                        "response_done_reason": resp_fields["done_reason"],
                        "response_done": resp_fields["done"],
                    }
                )
            except Exception as exc:
                async with self.lock:
                    self.failed_total += 1
                await self.add_log("ERROR", "Queued request failed", request_id=request_id, error=str(exc))
                await self.add_history_entry(
                    {
                        "request_id": request_id,
                        "method": item["method"],
                        "path": item["path"],
                        "client_key": item.get("client_key", "unknown"),
                        "client_label": item.get("client_label", "unknown"),
                        "client_ip": item.get("client_ip", "unknown"),
                        "client_ip_source": item.get("client_ip_source", "unknown"),
                        "client_kind": item.get("client_kind", "unknown"),
                        "client_details": item.get("client_details", ""),
                        "status_code": None,
                        "duration_ms": round((time.time() - started) * 1000, 2),
                        "enqueue_time": item.get("enqueue_time"),
                        "completed_at": now_iso(),
                        "question_preview": clip_text(make_body_preview(item["body"], item["headers"].get("content-type", ""))),
                        "answer_preview": clip_text(str(exc)),
                        "request_preview": make_body_preview(item["body"], item["headers"].get("content-type", "")),
                        "response_preview": "",
                        "request_details": detail_text(item["body"], item["headers"].get("content-type", "")),
                        "response_details": clip_text(str(exc), max_length=MAX_DETAIL_TEXT_LENGTH),
                        "request_model": str(item.get("request_model", "")),
                        "request_prompt": str(item.get("request_prompt", "")),
                        "response_model": "",
                        "response_text": "",
                        "response_created_at": "",
                        "response_done_reason": "",
                        "response_done": "",
                        "error": str(exc),
                    }
                )
                if not future.done():
                    future.set_exception(exc)
            finally:
                elapsed = round((time.time() - started) * 1000, 2)
                async with self.lock:
                    self.active -= 1
                    self.pending.pop(request_id, None)
                self.queue.task_done()
                await self.record_metric_snapshot()
                await self.add_log("INFO", "Queued request processed", request_id=request_id, duration_ms=elapsed)


state = MonitorState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for _ in range(state.workers):
        asyncio.create_task(state.process_queue())
    asyncio.create_task(model_refresher())
    await state.record_metric_snapshot()
    await state.add_log("INFO", "Monitor started", backend_url=state.backend_url, workers=state.workers)
    yield


app = FastAPI(title="SharedOllama Monitor", lifespan=lifespan)


async def model_refresher() -> None:
    while True:
        await state.refresh_models()
        await state.record_metric_snapshot()
        await asyncio.sleep(state.model_refresh_interval_seconds)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/")
async def root(request: Request) -> RedirectResponse:
    query = request.url.query
    target = "/monitor"
    if query:
        target = f"{target}?{query}"
    return RedirectResponse(url=target, status_code=307)


@app.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request) -> Response:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SharedOllama Monitor</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; background: #f7f8fa; color:#222; }
      .grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }
      .card { background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
      .toolbar { display:flex; gap:10px; margin: 12px 0 18px; flex-wrap: wrap; }
      .button { background:#0b5ed7; color:#fff; border:none; border-radius:6px; padding:8px 12px; cursor:pointer; text-decoration:none; font-size:13px; }
      .button.secondary { background:#6c757d; }
      .button.warn { background:#c97a00; }
      .button.danger { background:#c62828; }
      .hero-link { display:block; margin: 0 0 16px; background:#ffffff; border:1px solid #cfe0ff; border-radius:10px; padding:14px 16px; color:#0b3d91; text-decoration:none; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
      .hero-link strong { display:block; font-size:16px; margin-bottom:4px; }
      .hero-link span { font-size:13px; color:#35507a; }
      h1,h2 { margin: 0 0 10px; }
      pre { max-height: 240px; overflow: auto; background: #121417; color: #ffffff; padding: 10px; border-radius: 6px; }
      table { width: 100%; border-collapse: collapse; background:#fff; border-radius:8px; overflow:hidden; }
      th, td { text-align: left; border-bottom: 1px solid #eee; padding: 6px; font-size: 12px; }
      td.actions { white-space: nowrap; }
      .section { margin-top: 16px; }
      .details-link { color: #0b5ed7; text-decoration: none; font-weight: 600; }
      .details-link:hover { text-decoration: underline; }
      .state-badge { display:inline-block; border-radius:999px; padding:3px 8px; font-size:11px; font-weight:700; text-transform:uppercase; }
      .state-active { background:#dff3e4; color:#19692c; }
      .state-paused { background:#fff3cd; color:#8a5a00; }
      .state-blocked { background:#f8d7da; color:#842029; }
    .inline-note { margin: 8px 0 12px; padding: 10px 12px; background:#fff8e1; color:#7a5a00; border:1px solid #ecd9a2; border-radius:8px; font-size:12px; }
    </style>
  </head>
  <body>
    <h1>SharedOllama Monitor</h1>
    <a class="hero-link" id="graphHeroLink" target="_blank" rel="noopener noreferrer">
      <strong>Open Live Graph</strong>
      <span>Shows live totals for requests, queue, and completed jobs.</span>
    </a>
    <div class="toolbar">
      <a class="button" id="graphLink" target="_blank" rel="noopener noreferrer">Live graph</a>
      <button class="button secondary" onclick="refresh()">Refresh now</button>
    </div>
    <div class="grid" id="stats"></div>
    <div class="section">
      <h2>Loaded Models</h2>
      <pre id="models"></pre>
    </div>
    <div class="section">
      <h2>Client Controls</h2>
            <div class="inline-note">Observed IP may be a Docker or NAT peer. When a real client IP is provided through forwarding headers, the monitor prefers that value automatically.</div>
      <table>
                <thead><tr><th>Client</th><th>Type</th><th>Observed IP</th><th>IP Source</th><th>Queued</th><th>Done</th><th>Failed</th><th>State</th><th>Info</th><th>Actions</th></tr></thead>
        <tbody id="clients"></tbody>
      </table>
    </div>
    <div class="section">
      <h2>Queued Requests</h2>
      <table>
                                <caption style="text-align:left; padding:6px; font-size:12px;">Queued write requests waiting for Ollama processing. The IP column shows the observed IP inside the monitor and may be a Docker or NAT peer. Double-click a row to open full request details.</caption>
                <thead><tr><th>ID</th><th>Client</th><th>Observed IP</th><th>IP Source</th><th>Type</th><th>Method</th><th>Path</th><th>Queued At (Unix)</th><th>Body Preview</th></tr></thead>
        <tbody id="queue"></tbody>
      </table>
    </div>
        <div class="section">
            <h2>Completed Request History</h2>
            <table>
                                <caption style="text-align:left; padding:6px; font-size:12px;">Click Open details to inspect full request and response payload for a completed action in a new window. Observed IP may differ from the original remote host when Docker or NAT is in the path.</caption>
                                <thead><tr><th>ID</th><th>Client</th><th>Observed IP</th><th>IP Source</th><th>Type</th><th>Method</th><th>Path</th><th>Status</th><th>Duration ms</th><th>Completed At</th><th>Question</th><th>Answer</th><th>Action</th></tr></thead>
                <tbody id="history"></tbody>
            </table>
        </div>
    <div class="section">
      <h2>Alerts</h2>
      <pre id="alerts"></pre>
    </div>
    <div class="section">
      <h2>Errors</h2>
      <pre id="errors"></pre>
    </div>
    <div class="section">
      <h2>Logs</h2>
      <pre id="logs"></pre>
    </div>
    <script>
      document.getElementById('graphHeroLink').href = '/monitor/graph' + window.location.search;
      document.getElementById('graphLink').href = '/monitor/graph' + window.location.search;

      function stateBadge(state) {
        const span = document.createElement('span');
        const normalized = String(state || 'active').toLowerCase();
        span.className = 'state-badge state-' + normalized;
        span.textContent = normalized;
        return span;
      }

      function renderStats(stats) {
        const statsEl = document.getElementById('stats');
        statsEl.innerHTML = '';
        Object.entries(stats).forEach(([k, v]) => {
          const card = document.createElement('div');
          card.className = 'card';
          const label = document.createElement('b');
          label.textContent = k;
          const value = document.createElement('div');
          value.textContent = String(v);
          card.appendChild(label);
          card.appendChild(value);
          statsEl.appendChild(card);
        });
      }

      async function setClientState(clientKey, action) {
        const response = await fetch('/monitor/api/clients/' + encodeURIComponent(clientKey) + '/state' + window.location.search, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action })
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({ error: 'Unknown error' }));
          alert(payload.error || 'Failed updating client state');
          return;
        }
        await refresh();
      }

      function renderClients(clients) {
        const clientsEl = document.getElementById('clients');
        clientsEl.innerHTML = '';
        (Array.isArray(clients) ? clients : []).forEach((client) => {
          const tr = document.createElement('tr');

          [client.client_label, client.client_kind, client.client_ip, client.client_ip_source, client.queued_count, client.completed_count, client.failed_count].forEach((value) => {
            const td = document.createElement('td');
            td.textContent = String(value ?? '');
            tr.appendChild(td);
          });

          const stateTd = document.createElement('td');
          stateTd.appendChild(stateBadge(client.state));
          tr.appendChild(stateTd);

          const detailsTd = document.createElement('td');
          detailsTd.textContent = String(client.client_details ?? '');
          tr.appendChild(detailsTd);

          const actionsTd = document.createElement('td');
          actionsTd.className = 'actions';

          const pauseBtn = document.createElement('button');
          pauseBtn.className = 'button warn';
          pauseBtn.textContent = 'Pause';
          pauseBtn.onclick = () => setClientState(client.client_key, 'pause');
          actionsTd.appendChild(pauseBtn);

          const blockBtn = document.createElement('button');
          blockBtn.className = 'button danger';
          blockBtn.style.marginLeft = '6px';
          blockBtn.textContent = 'Stop';
          blockBtn.onclick = () => setClientState(client.client_key, 'block');
          actionsTd.appendChild(blockBtn);

          const resumeBtn = document.createElement('button');
          resumeBtn.className = 'button secondary';
          resumeBtn.style.marginLeft = '6px';
          resumeBtn.textContent = 'Resume';
          resumeBtn.onclick = () => setClientState(client.client_key, 'resume');
          actionsTd.appendChild(resumeBtn);

          tr.appendChild(actionsTd);
          clientsEl.appendChild(tr);
        });
      }

      function renderQueue(queue) {
        const queueEl = document.getElementById('queue');
        queueEl.innerHTML = '';
        queue.forEach((q) => {
          const tr = document.createElement('tr');
                    tr.title = 'Double-click to open details in a new window';
                    tr.style.cursor = 'pointer';
                    tr.addEventListener('dblclick', () => {
                        window.open(detailsUrl(q.request_id), '_blank', 'noopener,noreferrer');
                    });
          [q.request_id, q.client_label, q.client_ip, q.client_ip_source, q.client_kind, q.method, q.path, q.enqueue_time, q.body_preview].forEach((value) => {
            const td = document.createElement('td');
            td.textContent = String(value ?? '');
            tr.appendChild(td);
          });
          queueEl.appendChild(tr);
        });
      }

            function detailsUrl(requestId) {
                return '/monitor/details/' + encodeURIComponent(requestId) + window.location.search;
            }

            function renderHistory(history) {
                const historyEl = document.getElementById('history');
                historyEl.innerHTML = '';
                const items = Array.isArray(history) ? history : [];
                items.forEach((h) => {
                    const tr = document.createElement('tr');
                    [
                        h.request_id,
                        h.client_label,
                        h.client_ip,
                        h.client_ip_source,
                        h.client_kind,
                        h.method,
                        h.path,
                        h.status_code,
                        h.duration_ms,
                        h.completed_at,
                        h.question_preview,
                        h.answer_preview,
                    ].forEach((value) => {
                        const td = document.createElement('td');
                        td.textContent = String(value ?? '');
                        tr.appendChild(td);
                    });

                    const actionTd = document.createElement('td');
                    actionTd.className = 'actions';
                    const link = document.createElement('a');
                    link.className = 'details-link';
                    link.href = detailsUrl(h.request_id);
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.textContent = 'Open details';
                    actionTd.appendChild(link);
                    tr.appendChild(actionTd);

                    historyEl.appendChild(tr);
                });
            }

      async function refresh() {
        const r = await fetch('/monitor/api/state' + window.location.search);
        const s = await r.json();
        renderStats(s.stats);
        document.getElementById('models').textContent = JSON.stringify(s.models, null, 2);
        renderClients(s.clients);
        renderQueue(s.queue);
        renderHistory(s.history);
        document.getElementById('alerts').textContent = JSON.stringify(s.alerts, null, 2);
        document.getElementById('errors').textContent = JSON.stringify(s.errors, null, 2);
        document.getElementById('logs').textContent = JSON.stringify(s.logs, null, 2);
      }
      refresh();
      setInterval(refresh, 3000);
    </script>
  </body>
</html>
    """


@app.get("/monitor/api/state")
async def monitor_state(request: Request) -> Any:
    auth_error = await ensure_monitor_auth(request=request)
    if auth_error:
        return auth_error
    queue_items = await state.queue_snapshot()
    client_items = await state.client_snapshot()
    async with state.lock:
        stats = {
            "uptime_seconds": int(time.time() - state.started_at),
            "requests_total": state.requests_total,
            "requests_in_current_minute": state.requests_in_window,
            "rate_limit_per_minute": state.rate_limit_per_minute,
            "rate_limited_total": state.rate_limited_total,
            "queued_count": len(queue_items),
            "queue_max_size": state.max_queue_size,
            "active_workers": state.active,
            "workers": state.workers,
            "processed_total": state.processed_total,
            "failed_total": state.failed_total,
            "last_models_refresh": state.last_models_refresh,
        }
        return {
            "stats": stats,
            "models": state.models_cache,
            "queue": queue_items,
            "clients": client_items,
            "history": list(state.history),
            "logs": list(state.logs),
            "errors": list(state.errors),
            "alerts": list(state.alerts),
            "metrics_history": list(state.metric_history),
        }


@app.get("/monitor/api/clients")
async def client_list(request: Request) -> Any:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error
    return {"clients": await state.client_snapshot()}


@app.post("/monitor/api/clients/{client_key}/state")
async def update_client_state(request: Request, client_key: str) -> Any:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error

    payload = await request.json()
    action = str(payload.get("action", "")).strip().lower()
    if action not in {"pause", "block", "resume"}:
        return JSONResponse({"error": "invalid action"}, status_code=400)

    if action == "resume":
        updated = await state.set_client_state(client_key, "active")
        await state.add_log("INFO", "Client resumed", client_key=client_key)
        await state.record_metric_snapshot()
        return {"client": updated, "cancelled_queued": 0}

    next_state = "paused" if action == "pause" else "blocked"
    updated = await state.set_client_state(client_key, next_state)
    cancelled = 0
    if action == "block":
        cancelled = await state.cancel_pending_for_client(client_key, "Client is blocked")

    await state.add_log("WARN", "Client state changed", client_key=client_key, state=next_state, cancelled_queued=cancelled)
    await state.record_metric_snapshot()
    return {"client": updated, "cancelled_queued": cancelled}


@app.get("/monitor/graph", response_class=HTMLResponse)
@app.get("/monitor/Graph", response_class=HTMLResponse)
async def graph_page(request: Request) -> Response:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SharedOllama Live Graph</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; background:linear-gradient(180deg, #f4f6f9 0%, #eef4fb 100%); color:#1f2933; }
      h1 { margin: 0 0 8px; }
      p { margin: 0 0 16px; color:#52606d; }
      .layout { display:grid; grid-template-columns: 1fr; gap:14px; }
      .stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:10px; }
      .stat { background:#fff; border-radius:12px; padding:12px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border:1px solid #e4ebf3; }
      .stat-label { font-size:12px; color:#6b7a8c; margin-bottom:4px; }
      .stat-value { font-size:24px; font-weight:700; color:#102a43; }
      .wrap { background:#fff; border-radius:14px; padding:16px; box-shadow: 0 4px 18px rgba(16,42,67,0.08); border:1px solid #e4ebf3; }
      .chart-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-end; margin-bottom:12px; flex-wrap:wrap; }
      .chart-title { font-size:18px; font-weight:700; color:#102a43; }
      .chart-sub { font-size:12px; color:#6b7a8c; }
      canvas { width:100%; height:360px; border:1px solid #dde3ea; border-radius:12px; background:linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); }
      .legend { display:flex; gap:16px; margin-top:12px; flex-wrap:wrap; font-size:13px; }
      .axis-note { display:flex; justify-content:space-between; margin-top:10px; font-size:12px; color:#52606d; }
      .empty { margin-top:12px; padding:12px; border-radius:10px; background:#f7fafc; color:#52606d; font-size:13px; border:1px dashed #c7d2e2; }
      .legend span::before { content:''; display:inline-block; width:12px; height:12px; margin-right:6px; border-radius:2px; vertical-align:middle; }
      .req::before { background:#0b5ed7; }
      .queue::before { background:#c97a00; }
      .done::before { background:#198754; }
      @media (max-width: 640px) {
        body { margin: 12px; }
        canvas { height:300px; }
      }
    </style>
  </head>
  <body>
    <h1>Live Request Graph</h1>
    <p>Y-axis shows antall requests. X-axis shows de siste tidspunktene, med nyeste punkt til høyre.</p>
    <div class="layout">
      <div class="stats">
        <div class="stat"><div class="stat-label">Total requests</div><div class="stat-value" id="statRequests">0</div></div>
        <div class="stat"><div class="stat-label">Queue now</div><div class="stat-value" id="statQueue">0</div></div>
        <div class="stat"><div class="stat-label">Completed</div><div class="stat-value" id="statDone">0</div></div>
      </div>
      <div class="wrap">
      <div class="chart-head">
        <div>
          <div class="chart-title">Request Flow</div>
          <div class="chart-sub">Live snapshot of incoming, queued, and completed requests.</div>
        </div>
      </div>
      <canvas id="graph" width="1200" height="360"></canvas>
      <div class="axis-note">
        <span>Y: Number of requests</span>
        <span>X: Time, newest to the right</span>
      </div>
      <div class="legend">
        <span class="req">Total requests</span>
        <span class="queue">Queue</span>
        <span class="done">Completed</span>
      </div>
      <div class="empty" id="emptyState" style="display:none;">Waiting for more datapoints. The graph becomes clearer after a few requests or refresh cycles.</div>
      </div>
    </div>
    <script>
      const canvas = document.getElementById('graph');
      const ctx = canvas.getContext('2d');
      const padLeft = 64;
      const padRight = 28;
      const padTop = 24;
      const padBottom = 42;
      const statRequests = document.getElementById('statRequests');
      const statQueue = document.getElementById('statQueue');
      const statDone = document.getElementById('statDone');
      const emptyState = document.getElementById('emptyState');

      function setStats(points) {
        const last = points.length ? points[points.length - 1] : {};
        statRequests.textContent = String(Number(last.requests_total || 0));
        statQueue.textContent = String(Number(last.queued_count || 0));
        statDone.textContent = String(Number(last.processed_total || 0));
      }

      function drawLine(points, color, maxValue) {
        if (!points.length || maxValue <= 0) return;
        const w = canvas.width - padLeft - padRight;
        const h = canvas.height - padTop - padBottom;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        points.forEach((value, index) => {
          const x = padLeft + (w * index / Math.max(points.length - 1, 1));
          const y = padTop + h - (value / maxValue) * h;
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();

        points.forEach((value, index) => {
          const x = padLeft + (w * index / Math.max(points.length - 1, 1));
          const y = padTop + h - (value / maxValue) * h;
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      function drawArea(points, color, maxValue) {
        if (points.length < 2 || maxValue <= 0) return;
        const w = canvas.width - padLeft - padRight;
        const h = canvas.height - padTop - padBottom;
        ctx.beginPath();
        points.forEach((value, index) => {
          const x = padLeft + (w * index / Math.max(points.length - 1, 1));
          const y = padTop + h - (value / maxValue) * h;
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.lineTo(canvas.width - padRight, canvas.height - padBottom);
        ctx.lineTo(padLeft, canvas.height - padBottom);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }

      function drawAxes(maxValue, points) {
        const chartWidth = canvas.width - padLeft - padRight;
        const chartHeight = canvas.height - padTop - padBottom;
        const yTicks = 4;

        ctx.strokeStyle = '#d7dde5';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#52606d';
        ctx.font = '12px Arial';

        for (let i = 0; i <= yTicks; i++) {
          const ratio = i / yTicks;
          const y = padTop + chartHeight - (chartHeight * ratio);
          const value = Math.round(maxValue * ratio);
          ctx.beginPath();
          ctx.moveTo(padLeft, y);
          ctx.lineTo(canvas.width - padRight, y);
          ctx.stroke();
          ctx.fillText(String(value), 18, y + 4);
        }

        ctx.beginPath();
        ctx.moveTo(padLeft, padTop);
        ctx.lineTo(padLeft, canvas.height - padBottom);
        ctx.lineTo(canvas.width - padRight, canvas.height - padBottom);
        ctx.stroke();

        const labels = points.map((p) => new Date(p.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
        const xIndexes = [0, Math.floor((labels.length - 1) / 2), Math.max(labels.length - 1, 0)].filter((value, index, self) => self.indexOf(value) === index);
        xIndexes.forEach((index) => {
          const x = padLeft + (chartWidth * index / Math.max(labels.length - 1, 1));
          const label = labels[index] || '';
          ctx.fillText(label, x - 20, canvas.height - 14);
        });

        ctx.save();
        ctx.translate(16, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Requests', 0, 0);
        ctx.restore();
        ctx.fillText('Time', canvas.width / 2 - 12, canvas.height - 14);
      }

      function render(history) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const points = Array.isArray(history) ? history.slice(-60) : [];
        setStats(points);
        emptyState.style.display = points.length < 2 ? 'block' : 'none';
        const reqs = points.map((p) => Number(p.requests_total || 0));
        const queue = points.map((p) => Number(p.queued_count || 0));
        const done = points.map((p) => Number(p.processed_total || 0));
        const maxValue = Math.max(1, ...reqs, ...queue, ...done);
        drawAxes(maxValue, points);
        drawArea(done, 'rgba(25, 135, 84, 0.08)', maxValue);
        drawLine(reqs, '#0b5ed7', maxValue);
        drawLine(queue, '#c97a00', maxValue);
        drawLine(done, '#198754', maxValue);
      }

      async function refresh() {
        const response = await fetch('/monitor/api/state' + window.location.search);
        const payload = await response.json();
        render(payload.metrics_history || []);
      }

      refresh();
      setInterval(refresh, 3000);
    </script>
  </body>
</html>
    """


@app.get("/monitor/api/history")
async def history_list(request: Request) -> Any:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error
    return {"history": await state.history_snapshot()}


@app.get("/monitor/api/history/{request_id}")
async def history_details(request: Request, request_id: str) -> Any:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error
    entry = await state.history_item(request_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    return entry


@app.get("/monitor/details/{request_id}", response_class=HTMLResponse)
async def details_page(request: Request, request_id: str) -> Response:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error

    queue_entry: dict[str, Any] | None = None
    async with state.lock:
        pending = state.pending.get(request_id)
        if pending is not None:
            queue_entry = asdict(pending)

    history_entry = await state.history_item(request_id)

    if queue_entry is None and history_entry is None:
        return HTMLResponse("<h1>Details not found</h1>", status_code=404)

    source = "queue" if queue_entry is not None else "history"
    base = history_entry or queue_entry or {}

    method = base.get("method", "")
    path = base.get("path", "")
    client_label = base.get("client_label", "")
    client_ip = base.get("client_ip", "")
    client_ip_source = base.get("client_ip_source", "")
    client_kind = base.get("client_kind", "")
    client_details = base.get("client_details", "")
    status_code = base.get("status_code", "pending")
    duration_ms = base.get("duration_ms", "pending")
    enqueue_time = base.get("enqueue_time", "")
    completed_at = base.get("completed_at", "pending")

    request_details = ""
    response_details = ""
    request_model = str(base.get("request_model", ""))
    request_prompt = str(base.get("request_prompt", ""))
    request_stream = ""
    request_keep_alive = ""
    response_model = str(base.get("response_model", ""))
    response_text = str(base.get("response_text", ""))
    response_created_at = str(base.get("response_created_at", ""))
    response_done_reason = str(base.get("response_done_reason", ""))
    response_done = str(base.get("response_done", ""))

    if queue_entry:
        request_details = str(queue_entry.get("request_details", ""))
    if history_entry:
        request_details = str(history_entry.get("request_details", request_details))
        response_details = str(history_entry.get("response_details", ""))

    if not response_details:
        response_details = "Response is not available yet for this request."

    try:
        req_payload = json.loads(request_details)
    except Exception:
        req_payload = None

    if req_payload is not None:
        req_fields = extract_request_fields(req_payload)
        if not request_model:
            request_model = req_fields["model"]
        if not request_prompt:
            request_prompt = req_fields["prompt"]
        request_stream = req_fields["stream"]
        request_keep_alive = req_fields["keep_alive"]

    try:
        resp_payload = json.loads(response_details)
    except Exception:
        resp_payload = None

    if resp_payload is not None:
        resp_fields = extract_response_fields(resp_payload)
        if not response_model:
            response_model = resp_fields["model"]
        if not response_text:
            response_text = resp_fields["response"]
        if not response_created_at:
            response_created_at = resp_fields["created_at"]
        if not response_done_reason:
            response_done_reason = resp_fields["done_reason"]
        if not response_done:
            response_done = resp_fields["done"]

    request_details = escape(request_details)
    response_details = escape(response_details)

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Details {request_id}</title>
    <style>
            body {{ font-family: Arial, sans-serif; margin: 16px; background: #f7f8fa; color: #222; }}
            h1 {{ margin: 0 0 12px; }}
            .meta {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 10px; margin-bottom: 14px; }}
            .card {{ background: #fff; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
            .label {{ font-size: 12px; color: #5d6570; margin-bottom: 4px; }}
            .value {{ font-weight: 600; word-break: break-word; }}
            h2 {{ margin: 10px 0 6px; font-size: 16px; }}
                .payload-grid {{ display:grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap:10px; margin-bottom: 12px; }}
            textarea {{ width: 100%; min-height: 220px; resize: vertical; border: 1px solid #d7dde5; border-radius: 8px; padding: 10px; font-family: Consolas, monospace; font-size: 12px; background: #fff; color: #111; box-sizing: border-box; }}
    </style>
  </head>
  <body>
    <h1>Request Details</h1>
        <div class=\"meta\">
            <div class=\"card\"><div class=\"label\">request_id</div><div class=\"value\">{escape(request_id)}</div></div>
            <div class=\"card\"><div class=\"label\">source</div><div class=\"value\">{escape(str(source))}</div></div>
            <div class=\"card\"><div class=\"label\">method</div><div class=\"value\">{escape(str(method))}</div></div>
            <div class=\"card\"><div class=\"label\">path</div><div class=\"value\">{escape(str(path))}</div></div>
            <div class=\"card\"><div class=\"label\">client</div><div class=\"value\">{escape(str(client_label))}</div></div>
            <div class=\"card\"><div class=\"label\">observed_ip</div><div class=\"value\">{escape(str(client_ip))}</div></div>
            <div class=\"card\"><div class=\"label\">ip_source</div><div class=\"value\">{escape(str(client_ip_source))}</div></div>
            <div class=\"card\"><div class=\"label\">client_type</div><div class=\"value\">{escape(str(client_kind))}</div></div>
            <div class=\"card\"><div class=\"label\">status_code</div><div class=\"value\">{escape(str(status_code))}</div></div>
            <div class=\"card\"><div class=\"label\">duration_ms</div><div class=\"value\">{escape(str(duration_ms))}</div></div>
            <div class=\"card\"><div class=\"label\">enqueue_time</div><div class=\"value\">{escape(str(enqueue_time))}</div></div>
            <div class=\"card\"><div class=\"label\">completed_at</div><div class=\"value\">{escape(str(completed_at))}</div></div>
            <div class=\"card\"><div class=\"label\">client_details</div><div class=\"value\">{escape(str(client_details))}</div></div>
        </div>

        <h2>Request fields</h2>
        <div class="payload-grid">
            <div class="card"><div class="label">model</div><div class="value">{escape(request_model)}</div></div>
            <div class="card"><div class="label">stream</div><div class="value">{escape(request_stream or "")}</div></div>
            <div class="card"><div class="label">keep_alive</div><div class="value">{escape(request_keep_alive or "")}</div></div>
            <div class="card"><div class="label">prompt</div><div class="value">{escape(request_prompt)}</div></div>
        </div>

        <h2>Response fields</h2>
        <div class="payload-grid">
            <div class="card"><div class="label">model</div><div class="value">{escape(response_model)}</div></div>
            <div class="card"><div class="label">created_at</div><div class="value">{escape(response_created_at)}</div></div>
            <div class="card"><div class="label">done_reason</div><div class="value">{escape(response_done_reason)}</div></div>
            <div class="card"><div class="label">done</div><div class="value">{escape(response_done)}</div></div>
            <div class="card" style="grid-column: 1 / -1;"><div class="label">response</div><div class="value">{escape(response_text)}</div></div>
        </div>

        <h2>request_details:</h2>
        <textarea readonly>{request_details}</textarea>

        <h2>response_details:</h2>
        <textarea readonly>{response_details}</textarea>
  </body>
</html>
    """


@app.get("/monitor/api/queue")
async def queue_details(request: Request) -> Any:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error
    return {"queue": await state.queue_snapshot()}


@app.get("/monitor/api/models")
async def models(request: Request) -> Any:
    auth_error = await ensure_monitor_auth(request)
    if auth_error:
        return auth_error
    await state.refresh_models()
    async with state.lock:
        return {"models": state.models_cache, "last_refresh": state.last_models_refresh}


async def ensure_monitor_auth(request: Request | None) -> JSONResponse | None:
    if not state.monitor_token:
        return None
    if request is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    token = request.headers.get("x-monitor-token") or request.query_params.get("token", "")
    if token != state.monitor_token:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def redact_json(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for k, v in value.items():
            lowered = k.lower()
            if any(secret_key in lowered for secret_key in ("password", "token", "secret", "api_key", "authorization")):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = redact_json(v)
        return redacted
    if isinstance(value, list):
        return [redact_json(i) for i in value]
    return value


def make_body_preview(body: bytes, content_type: str) -> str:
    if not body:
        return ""
    if is_binary_payload(body, content_type):
        return f"[binary content omitted: type={content_type or 'unknown'}, bytes={len(body)}]"
    if "application/json" in content_type.lower():
        try:
            parsed = json.loads(body.decode("utf-8", errors="ignore"))
            redacted = redact_json(parsed)
            text = json.dumps(redacted, ensure_ascii=False)
            return text[:MAX_BODY_PREVIEW_LENGTH]
        except Exception:
            pass
    text = body.decode("utf-8", errors="ignore").replace("\n", " ")
    return text[:MAX_BODY_PREVIEW_LENGTH]


def clip_text(value: Any, max_length: int = MAX_HISTORY_PREVIEW_LENGTH) -> str:
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


def normalize_ip_candidate(value: str) -> str:
    token = str(value or "").strip().strip('"').strip("'")
    if not token:
        return ""

    # RFC 7239 Forwarded may contain key/value pairs like: for=192.168.1.62:1234;proto=http
    if token.lower().startswith("for="):
        token = token[4:].strip().strip('"').strip("'")
    if ";" in token:
        token = token.split(";", 1)[0].strip()

    if token.lower() in {"unknown", "_hidden"}:
        return ""

    # Bracketed IPv6, optionally with port: [2001:db8::1]:443
    if token.startswith("[") and "]" in token:
        token = token[1 : token.index("]")]

    # IPv4 with port.
    if "." in token and token.count(":") == 1:
        host, port = token.rsplit(":", 1)
        if port.isdigit():
            token = host

    try:
        return str(ip_address(token))
    except ValueError:
        return ""


def extract_remote_ip(request: Request) -> str:
    headers = {k.lower(): v for k, v in request.headers.items()}

    prioritized_headers = [
        "x-real-ip",
        "true-client-ip",
        "cf-connecting-ip",
        "x-client-ip",
    ]
    for header_name in prioritized_headers:
        normalized = normalize_ip_candidate(headers.get(header_name, ""))
        if normalized:
            return normalized

    xff = headers.get("x-forwarded-for", "")
    if xff:
        for candidate in xff.split(","):
            normalized = normalize_ip_candidate(candidate)
            if normalized:
                return normalized

    forwarded = headers.get("forwarded", "")
    if forwarded:
        for part in forwarded.split(","):
            for segment in part.split(";"):
                segment = segment.strip()
                if segment.lower().startswith("for="):
                    normalized = normalize_ip_candidate(segment)
                    if normalized:
                        return normalized

    if request.client and request.client.host:
        normalized = normalize_ip_candidate(request.client.host)
        if normalized:
            return normalized
    return "unknown"


def classify_client_kind(client_ip: str, headers: dict[str, str]) -> str:
    host = headers.get("host", "").lower()
    origin = headers.get("origin", "").lower()
    user_agent = headers.get("user-agent", "").lower()
    forwarded = headers.get("x-forwarded-for", "").lower()

    if "host.docker.internal" in host or "host.docker.internal" in origin:
        return "local-docker"
    if "docker" in user_agent or "container" in user_agent:
        return "docker-like"
    if forwarded:
        return "proxied"

    try:
        parsed = ip_address(client_ip)
    except ValueError:
        return "unknown"

    if parsed.is_loopback:
        return "loopback"
    if parsed.is_private:
        return "local-network"
    return "remote"


def build_client_info(request: Request) -> dict[str, str]:
    headers = {k.lower(): v for k, v in request.headers.items()}
    client_ip = extract_remote_ip(request)
    socket_ip = request.client.host if request.client and request.client.host else ""
    host = headers.get("host", "")
    origin = headers.get("origin", "")
    referer = headers.get("referer", "")
    user_agent = headers.get("user-agent", "")
    forwarded = headers.get("x-forwarded-for", "")
    real_ip = headers.get("x-real-ip", "")
    explicit_client_name = (
        headers.get("x-client-name")
        or headers.get("x-client-id")
        or headers.get("x-forwarded-host")
        or headers.get("x-forwarded-server")
        or ""
    )
    client_name = explicit_client_name.strip()
    if not client_name and user_agent:
        client_name = clip_text(user_agent, 80)

    client_kind = classify_client_kind(client_ip, headers)
    has_forwarded_identity = any(
        headers.get(header_name, "").strip()
        for header_name in ("x-real-ip", "true-client-ip", "cf-connecting-ip", "x-client-ip", "x-forwarded-for", "forwarded")
    )
    client_ip_source = "forwarded-header" if has_forwarded_identity else "socket-peer"
    display_ip = client_ip
    display_name = client_name.strip()
    if not has_forwarded_identity and client_ip == socket_ip and client_ip.startswith("172."):
        display_ip = f"{client_ip} (Docker peer)"
        client_ip_source = "socket-peer-nat-hidden"
        if display_name:
            display_name = f"{display_name} via NAT"

    label_parts = [display_name, display_ip.strip()]
    client_label = " | ".join(part for part in label_parts if part) or client_ip or "unknown client"
    key_suffix = explicit_client_name.strip().lower()
    client_key = f"{client_ip}|{key_suffix}" if key_suffix else client_ip

    detail_parts = [
        f"observed_ip={client_ip}",
        f"type={client_kind}",
        f"source={'forwarded-header' if has_forwarded_identity else 'socket-peer'}",
    ]
    if not has_forwarded_identity and socket_ip and client_ip == socket_ip:
        detail_parts.append("note=Real remote host IP is hidden by Docker/NAT unless x-forwarded-for or x-real-ip is sent")
    if client_name:
        detail_parts.append(f"name={client_name}")
    if host:
        detail_parts.append(f"host={host}")
    if origin:
        detail_parts.append(f"origin={origin}")
    if referer:
        detail_parts.append(f"referer={referer}")
    if forwarded:
        detail_parts.append(f"xff={forwarded}")
    if real_ip:
        detail_parts.append(f"x-real-ip={real_ip}")
    if socket_ip and socket_ip != client_ip:
        detail_parts.append(f"socket_ip={socket_ip}")
    if user_agent:
        detail_parts.append(f"ua={clip_text(user_agent, 120)}")

    return {
        "client_key": client_key,
        "client_label": clip_text(client_label, 140),
        "client_ip": client_ip,
        "client_ip_source": client_ip_source,
        "client_kind": client_kind,
        "client_name": client_name.strip(),
        "client_details": " | ".join(detail_parts),
    }


def safe_json_parse(body: bytes, content_type: str) -> Any | None:
    if not body:
        return None
    if "application/json" not in content_type.lower():
        return None
    try:
        return json.loads(body.decode("utf-8", errors="ignore"))
    except Exception:
        return None


def get_user_question(parsed_request: Any) -> str:
    if isinstance(parsed_request, dict):
        for key in ("question", "prompt", "input", "query"):
            if key in parsed_request and isinstance(parsed_request[key], str):
                return clip_text(parsed_request[key])

        messages = parsed_request.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        return clip_text(content)
    return ""


def get_model_answer(parsed_response: Any) -> str:
    if isinstance(parsed_response, dict):
        if isinstance(parsed_response.get("response"), str):
            return clip_text(parsed_response["response"])

        message = parsed_response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return clip_text(message["content"])

        if isinstance(parsed_response.get("content"), str):
            return clip_text(parsed_response["content"])
    return ""


def extract_request_fields(parsed_request: Any) -> dict[str, str]:
    model = ""
    prompt = ""
    stream = ""
    keep_alive = ""

    if isinstance(parsed_request, dict):
        if isinstance(parsed_request.get("model"), str):
            model = parsed_request["model"].strip()

        if isinstance(parsed_request.get("prompt"), str):
            prompt = parsed_request["prompt"].strip()
        elif isinstance(parsed_request.get("input"), str):
            prompt = parsed_request["input"].strip()
        else:
            prompt = get_user_question(parsed_request)

        if "stream" in parsed_request:
            stream = str(parsed_request.get("stream"))
        if "keep_alive" in parsed_request:
            keep_alive = str(parsed_request.get("keep_alive"))

    return {
        "model": clip_text(model, 240),
        "prompt": clip_text(prompt, 4000),
        "stream": clip_text(stream, 80),
        "keep_alive": clip_text(keep_alive, 80),
    }


def extract_response_fields(parsed_response: Any) -> dict[str, str]:
    model = ""
    response_text = ""
    created_at = ""
    done_reason = ""
    done = ""

    if isinstance(parsed_response, dict):
        if isinstance(parsed_response.get("model"), str):
            model = parsed_response["model"].strip()

        if isinstance(parsed_response.get("response"), str):
            response_text = parsed_response["response"]
        else:
            message = parsed_response.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                response_text = message["content"]
            elif isinstance(parsed_response.get("content"), str):
                response_text = parsed_response["content"]

        if "created_at" in parsed_response:
            created_at = str(parsed_response.get("created_at", ""))
        if "done_reason" in parsed_response:
            done_reason = str(parsed_response.get("done_reason", ""))
        if "done" in parsed_response:
            done = str(parsed_response.get("done", ""))

    return {
        "model": clip_text(model, 240),
        "response": clip_text(response_text, 4000),
        "created_at": clip_text(created_at, 120),
        "done_reason": clip_text(done_reason, 120),
        "done": clip_text(done, 40),
    }


def detail_text(body: bytes, content_type: str, omit_context: bool = False) -> str:
    if is_binary_payload(body, content_type):
        return f"Binary payload omitted from details view. content_type={content_type or 'unknown'}, bytes={len(body)}"

    parsed = safe_json_parse(body, content_type)
    if parsed is not None:
        redacted = redact_json(parsed)
        if omit_context and isinstance(redacted, dict) and "context" in redacted:
            redacted = dict(redacted)
            redacted.pop("context", None)
        return json.dumps(redacted, ensure_ascii=False, indent=2)[:MAX_DETAIL_TEXT_LENGTH]

    text = body.decode("utf-8", errors="ignore")
    return text[:MAX_DETAIL_TEXT_LENGTH]


def is_binary_payload(body: bytes, content_type: str) -> bool:
    lowered = content_type.lower()
    if any(lowered.startswith(prefix) for prefix in BINARY_CONTENT_PREFIXES):
        return True
    if "application/octet-stream" in lowered:
        return True
    if "application/pdf" in lowered:
        return True

    if not body:
        return False

    sample = body[:512]
    if b"\x00" in sample:
        return True

    non_text_bytes = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return non_text_bytes > len(sample) * 0.2


def extract_requested_model(path: str, parsed_request: Any) -> str | None:
    if not path.startswith("/api/") or path == "/api/pull":
        return None
    if not isinstance(parsed_request, dict):
        return None

    model = parsed_request.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def is_model_missing_response(status_code: int, content: bytes) -> bool:
    if status_code != 404:
        return False
    text = content.decode("utf-8", errors="ignore").lower()
    return "model" in text and ("not found" in text or "missing" in text)


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(full_path: str, request: Request) -> Response:
    path = "/" + full_path
    if path.startswith("/monitor"):
        return JSONResponse({"error": "not found"}, status_code=404)

    client_info = build_client_info(request)
    client_state = await state.get_client_state(client_info["client_key"])
    if client_state == "paused":
        await state.add_log("WARN", "Request rejected for paused client", path=path, client_key=client_info["client_key"])
        return JSONResponse({"error": "client is paused"}, status_code=423)
    if client_state == "blocked":
        await state.add_log("WARN", "Request rejected for blocked client", path=path, client_key=client_info["client_key"])
        return JSONResponse({"error": "client is blocked"}, status_code=403)

    allowed = await state.check_rate_limit()
    if not allowed:
        await state.add_log("WARN", "Rate limit exceeded", path=path, client_key=client_info["client_key"])
        return JSONResponse({"error": "rate limit exceeded"}, status_code=429)

    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "connection", "accept-encoding"}
    }
    request_id = str(uuid.uuid4())

    original_content_type = request.headers.get("content-type", "")
    request_json = safe_json_parse(body, original_content_type)
    requested_model = extract_requested_model(path, request_json)
    resolved_model: str | None = None

    if requested_model:
        resolved_model = state.resolve_model_name(requested_model)
        if not state.model_allowed(resolved_model):
            await state.add_log(
                "WARN",
                "Model blocked by allowlist",
                request_id=request_id,
                path=path,
                requested_model=requested_model,
                resolved_model=resolved_model,
            )
            return JSONResponse({"error": f"model '{resolved_model}' is not allowed"}, status_code=403)

        if isinstance(request_json, dict):
            request_json["model"] = resolved_model
            body = json.dumps(request_json, ensure_ascii=False).encode("utf-8")
            headers["content-type"] = "application/json"

        if resolved_model != requested_model:
            await state.add_log(
                "INFO",
                "Model alias resolved",
                request_id=request_id,
                requested_model=requested_model,
                resolved_model=resolved_model,
            )

        if not await state.ensure_model_available(resolved_model, request_id=request_id):
            return JSONResponse({"error": f"model '{resolved_model}' is not available"}, status_code=404)

    request_fields = extract_request_fields(request_json)
    effective_content_type = headers.get("content-type", original_content_type)

    should_queue_request = request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"} and path.startswith("/api/")
    if not should_queue_request:
        try:
            attempts = 1 + (state.model_pull_retry_count if resolved_model else 0)
            backend_resp: httpx.Response | None = None

            for attempt in range(attempts):
                async with httpx.AsyncClient(timeout=state.upstream_timeout) as client:
                    backend_resp = await client.request(
                        request.method, f"{state.backend_url}{path}", content=body, headers=headers
                    )

                if (
                    resolved_model
                    and attempt < attempts - 1
                    and is_model_missing_response(backend_resp.status_code, backend_resp.content)
                ):
                    await state.add_log(
                        "WARN",
                        "Model not found at upstream, retrying after pull",
                        request_id=request_id,
                        model=resolved_model,
                        attempt=attempt + 1,
                    )
                    if not await state.ensure_model_available(resolved_model, request_id=request_id):
                        break
                    continue
                break

            if backend_resp is None:
                return JSONResponse({"error": "upstream unavailable"}, status_code=502)

            return Response(
                content=backend_resp.content,
                status_code=backend_resp.status_code,
                media_type=backend_resp.headers.get("content-type"),
            )
        except Exception as exc:
            await state.add_log("ERROR", "Proxy request failed", path=path, error=str(exc))
            return JSONResponse({"error": "upstream unavailable"}, status_code=502)

    if state.queue.full():
        await state.add_alert("Queue capacity reached", queue_size=state.queue.qsize(), max_size=state.max_queue_size)
        return JSONResponse({"error": "queue full"}, status_code=429)

    preview = make_body_preview(body, effective_content_type)
    item_info = QueueItemInfo(
        request_id=request_id,
        method=request.method.upper(),
        path=path,
        enqueue_time=time.time(),
        body_preview=preview,
        request_details=detail_text(body, effective_content_type),
        request_model=request_fields["model"],
        request_prompt=request_fields["prompt"],
        client_key=client_info["client_key"],
        client_label=client_info["client_label"],
        client_ip=client_info["client_ip"],
        client_ip_source=client_info["client_ip_source"],
        client_kind=client_info["client_kind"],
        client_details=client_info["client_details"],
    )
    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[dict[str, Any]] = loop.create_future()

    async with state.lock:
        state.pending[request_id] = item_info

    await state.queue.put(
        {
            "request_id": request_id,
            "method": request.method.upper(),
            "path": path,
            "body": body,
            "headers": headers,
            "future": result_future,
            "enqueue_time": item_info.enqueue_time,
            "request_model": item_info.request_model,
            "request_prompt": item_info.request_prompt,
            "client_key": item_info.client_key,
            "client_label": item_info.client_label,
            "client_ip": item_info.client_ip,
            "client_ip_source": item_info.client_ip_source,
            "client_kind": item_info.client_kind,
            "client_details": item_info.client_details,
        }
    )

    if state.queue.qsize() >= state.alert_queue_threshold:
        await state.add_alert(
            "Queue threshold reached",
            queue_size=state.queue.qsize(),
            threshold=state.alert_queue_threshold,
        )

    await state.record_metric_snapshot()
    await state.add_log(
        "INFO",
        "Queued request accepted",
        request_id=request_id,
        path=path,
        queue_size=state.queue.qsize(),
        client_key=item_info.client_key,
        client_ip=item_info.client_ip,
    )

    try:
        result = await result_future
        return Response(
            content=result["content"],
            status_code=result["status_code"],
            media_type=result["headers"].get("content-type"),
        )
    except Exception as exc:
        await state.add_log("ERROR", "Queued response failed", request_id=request_id, error=str(exc))
        return JSONResponse({"error": "upstream unavailable"}, status_code=502)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
