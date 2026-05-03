"""Admin and Monitor UI for SharedOllama. Runs on separate port (11444)."""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from html import escape

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .shared import (
    DEFAULT_ADMIN_PORT,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_PORT,
    MonitorState,
    check_backend_version,
    clip_text,
    ensure_monitor_auth,
    now_iso,
    parse_port,
    run_ollama_control,
)

state = MonitorState()


async def _proxy_monitor_request(path: str, method: str = "GET", payload: dict | None = None) -> Response:
  runtime = await state.runtime_config_snapshot()
  proxy_port = parse_port(runtime.get("shared_port", 11434), 11434)
  url = f"http://127.0.0.1:{proxy_port}{path}"
  headers: dict[str, str] = {}
  if state.monitor_token:
    headers["x-monitor-token"] = state.monitor_token

  try:
    async with httpx.AsyncClient(timeout=6.0) as client:
      response = await client.request(method=method, url=url, headers=headers, json=payload)
  except Exception as exc:
    return JSONResponse({"error": f"proxy monitor unavailable: {exc}"}, status_code=502)

  content_type = response.headers.get("content-type", "")
  if "application/json" in content_type.lower():
    try:
      return JSONResponse(response.json(), status_code=response.status_code)
    except Exception:
      return JSONResponse({"error": "invalid proxy monitor response"}, status_code=502)

  return JSONResponse({"error": "unexpected proxy monitor response"}, status_code=502)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Refresh models in background without blocking startup
    try:
        await asyncio.wait_for(state.refresh_models(), timeout=3.0)
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass
    try:
        yield
    finally:
        pass


app = FastAPI(title="SharedOllama Admin", lifespan=lifespan)


@app.get("/monitor", response_class=HTMLResponse)
@app.get("/Monitor", response_class=HTMLResponse)
async def monitor_page(request: Request) -> Response:
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
    if auth_error:
        return auth_error

    return HTMLResponse("""<!doctype html>
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
      .button.danger { background:#dc3545; }
      .section { margin-top: 20px; padding-bottom: 12px; border-bottom: 1px solid #ddd; }
      h1, h2 { margin-top: 0; }
      pre { background: #f4f5f7; padding: 12px; border-radius: 8px; overflow-x: auto; max-height: 280px; font-size: 12px; }
      table { width: 100%; border-collapse: collapse; margin: 10px 0; }
      th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; font-size: 12px; }
      th { background: #f8f9fa; font-weight: 600; }
      .state-active { background:#dff3e4; color:#19692c; }
      .state-paused { background:#fff3cd; color:#8a5a00; }
      .state-blocked { background:#f8d7da; color:#842029; }
      .inline-note { margin: 8px 0 12px; padding: 10px 12px; background:#fff8e1; color:#7a5a00; border:1px solid #ecd9a2; border-radius:8px; font-size:12px; }
      .strategy-note { margin: 8px 0 12px; padding: 10px 12px; background:#e9f2ff; color:#184a8c; border:1px solid #bfd6ff; border-radius:8px; font-size:12px; }
      .advanced-box { margin: 8px 0 10px; border: 1px solid #d6dce3; border-radius: 8px; background: #fff; }
      .advanced-box summary { cursor: pointer; padding: 10px 12px; font-size: 12px; color: #4a5562; font-weight: 600; }
      .advanced-content { padding: 0 10px 8px; }
      .admin-grid { display:grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap:10px; margin:8px 0 10px; }
      .form-row { display:flex; flex-direction:column; gap:4px; }
      .form-row label { font-size:12px; color:#4a5562; }
      .form-row input { padding:7px; border:1px solid #d6dce3; border-radius:6px; font-size:13px; }
      .admin-actions { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0; }
      .admin-status { font-size:12px; color:#4a5562; margin-bottom:8px; }
      .admin-output { max-height: 180px; }
      @media (max-width: 820px) {
        .admin-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <h1>SharedOllama Monitor</h1>
    <div class="toolbar">
      <button class="button" onclick="refresh()">Refresh now</button>
    </div>
    <div class="grid" id="stats"></div>
    <div class="section">
      <h2>Admin: WSL Ollama Setup</h2>
      <div class="inline-note">This section controls local WSL Ollama. Use Start/Restart to run Ollama on the configured host and port.</div>
      <div class="strategy-note">Strategy: Client URL is the public ingress your apps connect to. Backend URL is internal and should normally point to WSL Ollama on a different port.</div>
      <div class="admin-status" id="adminStatus"></div>
      <div class="admin-grid">
        <div class="form-row">
          <label for="clientUrl">Client URL (connect clients here)</label>
          <input id="clientUrl" type="text" readonly />
        </div>
        <div class="form-row">
          <label for="clientPort">Client port</label>
          <input id="clientPort" type="number" readonly />
        </div>
      </div>
      <details class="advanced-box">
        <summary>Advanced: Internal backend settings</summary>
        <div class="advanced-content">
          <div class="admin-grid">
        <div class="form-row">
          <label for="backendUrl">Internal backend URL (proxy target)</label>
          <input id="backendUrl" type="text" placeholder="http://127.0.0.1:11435" />
        </div>
        <div class="form-row">
          <label for="ollamaHost">Internal backend host</label>
          <input id="ollamaHost" type="text" placeholder="127.0.0.1" />
        </div>
        <div class="form-row">
          <label for="ollamaPort">Internal backend port</label>
          <input id="ollamaPort" type="number" min="1" max="65535" placeholder="11435" />
        </div>
          </div>
        </div>
      </details>
      <div class="admin-actions">
        <button class="button" onclick="saveAdminConfig()">Save config</button>
        <button class="button secondary" onclick="refreshAdmin()">Refresh admin state</button>
        <button class="button" onclick="runOllamaAction('start')">Start Ollama</button>
        <button class="button warn" onclick="runOllamaAction('restart')">Restart Ollama</button>
        <button class="button danger" onclick="runOllamaAction('stop')">Stop Ollama</button>
        <button class="button secondary" onclick="runOllamaAction('status')">Check status</button>
      </div>
      <pre id="adminOutput" class="admin-output"></pre>
    </div>
    <div class="section">
      <h2>Loaded Models</h2>
      <pre id="models"></pre>
    </div>
    <div class="section">
      <h2>Queue</h2>
      <table><thead><tr><th>request_id</th><th>client_label</th><th>client_ip</th><th>ip_source</th><th>client_kind</th><th>priority</th><th>method</th><th>path</th><th>enqueue_time</th><th>body_preview</th></tr></thead><tbody id="queue"></tbody></table>
    </div>
    <div class="section">
      <h2>Clients</h2>
      <table><thead><tr><th>client_label</th><th>kind</th><th>ip</th><th>ip_source</th><th>priority</th><th>queued</th><th>completed</th><th>failed</th><th>state</th><th>details</th><th>actions</th></tr></thead><tbody id="clients"></tbody></table>
    </div>
    <div class="section">
      <h2>History (Recent)</h2>
      <table><thead><tr><th>request_id</th><th>client_label</th><th>client_ip</th><th>ip_source</th><th>kind</th><th>method</th><th>path</th><th>status</th><th>duration_ms</th><th>completed_at</th><th>question</th><th>answer</th><th>action</th></tr></thead><tbody id="history"></tbody></table>
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
      function statBadge(text, bgColor) {
        const el = document.createElement('span');
        el.textContent = text;
        el.className = 'state-' + text.toLowerCase();
        return el;
      }

      function renderStats(stats) {
        const statsEl = document.getElementById('stats');
        statsEl.innerHTML = '';
        const cards = [
          { label: 'Uptime', value: stats.uptime_seconds + 's' },
          { label: 'Total requests', value: stats.requests_total },
          { label: 'Rate limit', value: stats.rate_limit_per_minute + '/min' },
          { label: 'Rate limited', value: stats.rate_limited_total },
          { label: 'Queued', value: stats.queued_count + ' / ' + stats.queue_max_size },
          { label: 'Active workers', value: stats.active_workers + ' / ' + stats.workers },
          { label: 'Processed', value: stats.processed_total },
          { label: 'Failed', value: stats.failed_total },
          { label: 'Backend target', value: stats.backend_url },
        ];
        cards.forEach((card) => {
          const el = document.createElement('div');
          el.className = 'card';
          el.innerHTML = `<div style="font-size:12px; color:#6b7a8c;">${card.label}</div><div style="font-weight:600; font-size:16px;">${card.value}</div>`;
          statsEl.appendChild(el);
        });
      }

      function writeAdminStatus(state) {
        const el = document.getElementById('adminStatus');
        if (!state) {
          el.textContent = 'Admin state unavailable';
          return;
        }
        const runtime = state.runtime_config || {};
        const client = state.client_endpoint || {};
        const backend = state.backend || {};
        const ollama = state.ollama || {};
        el.textContent = 'Client endpoint (apps use this): ' + String(client.url || '') +
          ' | Internal backend target: ' + String(runtime.backend_url || '') +
          ' | Backend reachable: ' + String(Boolean(backend.ok)) +
          ' | Ollama control OK: ' + String(Boolean(ollama.ok));
      }

      function renderAdmin(state) {
        const runtime = (state && state.runtime_config) || {};
        const client = (state && state.client_endpoint) || {};
        document.getElementById('clientUrl').value = String(client.url || 'http://127.0.0.1:11434');
        document.getElementById('clientPort').value = String(client.port || runtime.shared_port || '11434');
        document.getElementById('backendUrl').value = String(runtime.backend_url || '');
        document.getElementById('ollamaHost').value = String(runtime.ollama_host || '127.0.0.1');
        document.getElementById('ollamaPort').value = String(runtime.ollama_port || '11434');
        writeAdminStatus(state);
        document.getElementById('adminOutput').textContent = JSON.stringify(state, null, 2);
      }

      async function refreshAdmin() {
        const response = await fetch('/monitor/api/admin/state' + window.location.search);
        const payload = await response.json();
        renderAdmin(payload);
      }

      async function saveAdminConfig() {
        const payload = {
          backend_url: document.getElementById('backendUrl').value,
          ollama_host: document.getElementById('ollamaHost').value,
          ollama_port: Number(document.getElementById('ollamaPort').value)
        };
        const response = await fetch('/monitor/api/admin/config' + window.location.search, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        document.getElementById('adminOutput').textContent = JSON.stringify(data, null, 2);
        if (!response.ok) {
          alert(data.error || 'Failed saving admin config');
          return;
        }
        await refreshAdmin();
        await refresh();
      }

      async function runOllamaAction(action) {
        const response = await fetch('/monitor/api/admin/ollama/' + encodeURIComponent(action) + window.location.search, {
          method: 'POST'
        });
        const data = await response.json();
        document.getElementById('adminOutput').textContent = JSON.stringify(data, null, 2);
        if (!response.ok) {
          alert(data.error || ('Failed action: ' + action));
          return;
        }
        await refreshAdmin();
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
          [client.client_label, client.client_kind, client.client_ip, client.client_ip_source, client.priority, client.queued_count, client.completed_count, client.failed_count].forEach((value) => {
            const td = document.createElement('td');
            td.textContent = String(value ?? '');
            tr.appendChild(td);
          });
          const stateTd = document.createElement('td');
          stateTd.appendChild(statBadge(client.state));
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
          [q.request_id, q.client_label, q.client_ip, q.client_ip_source, q.client_kind, q.client_priority, q.method, q.path, q.enqueue_time, q.body_preview].forEach((value) => {
            const td = document.createElement('td');
            td.textContent = String(value ?? '');
            tr.appendChild(td);
          });
          queueEl.appendChild(tr);
        });
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
      refreshAdmin();
      setInterval(refresh, 3000);
      setInterval(refreshAdmin, 10000);
    </script>
  </body>
</html>
""")


@app.get("/monitor/api/state")
async def monitor_state(request: Request):
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
    if auth_error:
        return auth_error
    return await _proxy_monitor_request("/monitor/api/state")


@app.get("/monitor/api/admin/state")
async def admin_state(request: Request):
  auth_error = await ensure_monitor_auth(request, state.monitor_token)
  if auth_error:
    return auth_error

  runtime = await state.runtime_config_snapshot()
  backend = await check_backend_version(runtime.get("backend_url", state.backend_url))
  client_port = parse_port(runtime.get("shared_port", 11434), 11434)
  client_endpoint = {
    "url": f"http://127.0.0.1:{client_port}",
    "host": "127.0.0.1",
    "port": client_port,
    "note": "Client applications should connect to this endpoint.",
  }
  ollama_status = await run_ollama_control(
    action="status",
    host=str(runtime.get("ollama_host", DEFAULT_OLLAMA_HOST)),
    port=parse_port(runtime.get("ollama_port", DEFAULT_OLLAMA_PORT), DEFAULT_OLLAMA_PORT),
  )

  return {
    "runtime_config": runtime,
    "client_endpoint": client_endpoint,
    "backend": backend,
    "ollama": ollama_status,
  }


@app.post("/monitor/api/admin/config")
async def admin_update_config(request: Request):
  auth_error = await ensure_monitor_auth(request, state.monitor_token)
  if auth_error:
    return auth_error

  payload = await request.json()
  host = str(payload.get("ollama_host", DEFAULT_OLLAMA_HOST)).strip()
  port = parse_port(payload.get("ollama_port", DEFAULT_OLLAMA_PORT), DEFAULT_OLLAMA_PORT)
  backend_url = str(payload.get("backend_url", "")).strip().rstrip("/") or f"http://{host}:{port}"

  saved = await state.update_runtime_config(
    backend_url=backend_url,
    shared_port=11434,
    ollama_host=host,
    ollama_port=port,
  )
  await state.mark_admin_action("config_saved", backend_url=backend_url)
  return {"ok": True, "config": saved}


@app.post("/monitor/api/admin/ollama/{action}")
async def admin_ollama_action(request: Request, action: str):
  auth_error = await ensure_monitor_auth(request, state.monitor_token)
  if auth_error:
    return auth_error

  normalized_action = action.strip().lower()
  if normalized_action not in {"start", "stop", "restart", "status"}:
    return JSONResponse({"error": "invalid action"}, status_code=400)

  runtime = await state.runtime_config_snapshot()
  result = await run_ollama_control(
    action=normalized_action,
    host=str(runtime.get("ollama_host", DEFAULT_OLLAMA_HOST)),
    port=parse_port(runtime.get("ollama_port", DEFAULT_OLLAMA_PORT), DEFAULT_OLLAMA_PORT),
  )
  await state.mark_admin_action(
    f"ollama_{normalized_action}",
    ok=result.get("ok", False),
    exit_code=result.get("exit_code"),
  )
  status_code = 200 if result.get("ok") else 500
  return JSONResponse({"ok": bool(result.get("ok")), "action": normalized_action, "result": result}, status_code=status_code)


@app.get("/monitor/api/clients")
async def client_list(request: Request):
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
    if auth_error:
        return auth_error
    return await _proxy_monitor_request("/monitor/api/clients")


@app.post("/monitor/api/clients/{client_key}/state")
async def update_client_state(request: Request, client_key: str):
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
    if auth_error:
        return auth_error
    payload = await request.json()
    return await _proxy_monitor_request(
        path=f"/monitor/api/clients/{client_key}/state",
        method="POST",
        payload={"action": payload.get("action", "")},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_ADMIN_PORT)
