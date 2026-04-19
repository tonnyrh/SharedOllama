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

                request_preview = make_body_preview(item["body"], req_content_type)
                response_preview = make_body_preview(resp.content, resp_content_type)
                question_preview = get_user_question(req_parsed) or clip_text(request_preview)
                answer_preview = get_model_answer(resp_parsed) or clip_text(response_preview)

                await self.add_history_entry(
                    {
                        "request_id": request_id,
                        "method": item["method"],
                        "path": item["path"],
                        "status_code": resp.status_code,
                        "duration_ms": round((time.time() - started) * 1000, 2),
                        "enqueue_time": item.get("enqueue_time"),
                        "completed_at": now_iso(),
                        "question_preview": question_preview,
                        "answer_preview": answer_preview,
                        "request_preview": request_preview,
                        "response_preview": response_preview,
                        "request_details": detail_text(item["body"], req_content_type),
                        "response_details": detail_text(resp.content, resp_content_type),
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
                await self.add_log("INFO", "Queued request processed", request_id=request_id, duration_ms=elapsed)


state = MonitorState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for _ in range(state.workers):
        asyncio.create_task(state.process_queue())
    asyncio.create_task(model_refresher())
    await state.add_log("INFO", "Monitor started", backend_url=state.backend_url, workers=state.workers)
    yield


app = FastAPI(title="SharedOllama Monitor", lifespan=lifespan)


async def model_refresher() -> None:
    while True:
        await state.refresh_models()
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
      h1,h2 { margin: 0 0 10px; }
      pre { max-height: 240px; overflow: auto; background: #121417; color: #ffffff; padding: 10px; border-radius: 6px; }
      table { width: 100%; border-collapse: collapse; background:#fff; border-radius:8px; overflow:hidden; }
      th, td { text-align: left; border-bottom: 1px solid #eee; padding: 6px; font-size: 12px; }
    td.actions { white-space: nowrap; }
      .section { margin-top: 16px; }
    .details-link { color: #0b5ed7; text-decoration: none; font-weight: 600; }
    .details-link:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <h1>SharedOllama Monitor</h1>
    <div class="grid" id="stats"></div>
    <div class="section">
      <h2>Loaded Models</h2>
      <pre id="models"></pre>
    </div>
    <div class="section">
      <h2>Queued Requests</h2>
      <table>
                <caption style="text-align:left; padding:6px; font-size:12px;">Queued write requests waiting for Ollama processing. Double-click a row to open full request details.</caption>
        <thead><tr><th>ID</th><th>Method</th><th>Path</th><th>Queued At (Unix)</th><th>Body Preview</th></tr></thead>
        <tbody id="queue"></tbody>
      </table>
    </div>
        <div class="section">
            <h2>Completed Request History</h2>
            <table>
                <caption style="text-align:left; padding:6px; font-size:12px;">Click Open details to inspect full request and response payload for a completed action in a new window.</caption>
                <thead><tr><th>ID</th><th>Method</th><th>Path</th><th>Status</th><th>Duration ms</th><th>Completed At</th><th>Question</th><th>Answer</th><th>Action</th></tr></thead>
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
          [q.request_id, q.method, q.path, q.enqueue_time, q.body_preview].forEach((value) => {
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
            "history": list(state.history),
            "logs": list(state.logs),
            "errors": list(state.errors),
            "alerts": list(state.alerts),
        }


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
    status_code = base.get("status_code", "pending")
    duration_ms = base.get("duration_ms", "pending")
    enqueue_time = base.get("enqueue_time", "")
    completed_at = base.get("completed_at", "pending")

    request_details = ""
    response_details = ""

    if queue_entry:
        request_details = str(queue_entry.get("request_details", ""))
    if history_entry:
        request_details = str(history_entry.get("request_details", request_details))
        response_details = str(history_entry.get("response_details", ""))

    if not response_details:
        response_details = "Response is not available yet for this request."

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
            <div class=\"card\"><div class=\"label\">status_code</div><div class=\"value\">{escape(str(status_code))}</div></div>
            <div class=\"card\"><div class=\"label\">duration_ms</div><div class=\"value\">{escape(str(duration_ms))}</div></div>
            <div class=\"card\"><div class=\"label\">enqueue_time</div><div class=\"value\">{escape(str(enqueue_time))}</div></div>
            <div class=\"card\"><div class=\"label\">completed_at</div><div class=\"value\">{escape(str(completed_at))}</div></div>
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


def detail_text(body: bytes, content_type: str) -> str:
    if is_binary_payload(body, content_type):
        return f"Binary payload omitted from details view. content_type={content_type or 'unknown'}, bytes={len(body)}"

    parsed = safe_json_parse(body, content_type)
    if parsed is not None:
        return json.dumps(redact_json(parsed), ensure_ascii=False, indent=2)[:MAX_DETAIL_TEXT_LENGTH]

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

    allowed = await state.check_rate_limit()
    if not allowed:
        await state.add_log("WARN", "Rate limit exceeded", path=path)
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
        }
    )

    if state.queue.qsize() >= state.alert_queue_threshold:
        await state.add_alert(
            "Queue threshold reached",
            queue_size=state.queue.qsize(),
            threshold=state.alert_queue_threshold,
        )

    await state.add_log("INFO", "Queued request accepted", request_id=request_id, path=path, queue_size=state.queue.qsize())

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
