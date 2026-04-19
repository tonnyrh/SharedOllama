import asyncio
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

MAX_BODY_PREVIEW_LENGTH = 200


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QueueItemInfo:
    request_id: str
    method: str
    path: str
    enqueue_time: float
    body_preview: str


class MonitorState:
    def __init__(self) -> None:
        self.backend_url = os.getenv("OLLAMA_BACKEND_URL", "http://ollama:11434").rstrip("/")
        self.rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "200"))
        self.workers = int(os.getenv("QUEUE_WORKERS", "2"))
        self.max_log_entries = int(os.getenv("MAX_LOG_ENTRIES", "500"))
        self.max_error_entries = int(os.getenv("MAX_ERROR_ENTRIES", "200"))
        self.max_alert_entries = int(os.getenv("MAX_ALERT_ENTRIES", "200"))
        self.alert_queue_threshold = int(os.getenv("ALERT_QUEUE_THRESHOLD", "50"))
        self.upstream_timeout = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "300"))
        self.monitor_token = os.getenv("MONITOR_TOKEN", "")

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
        self.models_cache: list[dict[str, Any]] = []
        self.last_models_refresh: str | None = None

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
            except Exception as exc:
                async with self.lock:
                    self.failed_total += 1
                await self.add_log("ERROR", "Queued request failed", request_id=request_id, error=str(exc))
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
        await asyncio.sleep(60)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


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
      .section { margin-top: 16px; }
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
        <caption style="text-align:left; padding:6px; font-size:12px;">Queued write requests waiting for Ollama processing, including request metadata and redacted preview payloads.</caption>
        <thead><tr><th>ID</th><th>Method</th><th>Path</th><th>Queued At (Unix)</th><th>Body Preview</th></tr></thead>
        <tbody id="queue"></tbody>
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
          [q.request_id, q.method, q.path, q.enqueue_time, q.body_preview].forEach((value) => {
            const td = document.createElement('td');
            td.textContent = String(value ?? '');
            tr.appendChild(td);
          });
          queueEl.appendChild(tr);
        });
      }

      async function refresh() {
        const r = await fetch('/monitor/api/state' + window.location.search);
        const s = await r.json();
        renderStats(s.stats);
        document.getElementById('models').textContent = JSON.stringify(s.models, null, 2);
        renderQueue(s.queue);
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
            "logs": list(state.logs),
            "errors": list(state.errors),
            "alerts": list(state.alerts),
        }


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

    should_queue_request = request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"} and path.startswith("/api/")
    if not should_queue_request:
        try:
            async with httpx.AsyncClient(timeout=state.upstream_timeout) as client:
                backend_resp = await client.request(
                    request.method, f"{state.backend_url}{path}", content=body, headers=headers
                )
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

    request_id = str(uuid.uuid4())
    preview = make_body_preview(body, request.headers.get("content-type", ""))
    item_info = QueueItemInfo(
        request_id=request_id,
        method=request.method.upper(),
        path=path,
        enqueue_time=time.time(),
        body_preview=preview,
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
