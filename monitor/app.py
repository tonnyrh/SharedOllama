"""Pure Ollama proxy for SharedOllama. Runs on standard port (11434)."""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from ipaddress import ip_address
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .shared import MonitorState, QueueItemInfo, ensure_monitor_auth, now_iso, parse_priority_value

state = MonitorState()


def _normalize_ip_candidate(value: str) -> str:
    token = str(value or "").strip().strip('"').strip("'")
    if not token:
        return ""
    if token.lower().startswith("for="):
        token = token[4:].strip().strip('"').strip("'")
    if ";" in token:
        token = token.split(";", 1)[0].strip()
    if token.lower() in {"unknown", "_hidden"}:
        return ""
    if token.startswith("[") and "]" in token:
        token = token[1 : token.index("]")]
    if "." in token and token.count(":") == 1:
        host, port = token.rsplit(":", 1)
        if port.isdigit():
            token = host
    try:
        return str(ip_address(token))
    except ValueError:
        return ""


def _extract_remote_ip(request: Request) -> str:
    headers = {k.lower(): v for k, v in request.headers.items()}
    for header_name in ["x-real-ip", "true-client-ip", "cf-connecting-ip", "x-client-ip"]:
        normalized = _normalize_ip_candidate(headers.get(header_name, ""))
        if normalized:
            return normalized
    xff = headers.get("x-forwarded-for", "")
    if xff:
        for candidate in xff.split(","):
            normalized = _normalize_ip_candidate(candidate)
            if normalized:
                return normalized
    if request.client and request.client.host:
        normalized = _normalize_ip_candidate(request.client.host)
        if normalized:
            return normalized
    return "unknown"


def _classify_client_kind(client_ip: str, headers: dict[str, str]) -> str:
    user_agent = headers.get("user-agent", "").lower()
    forwarded = headers.get("x-forwarded-for", "").lower()
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


def _build_client_info(request: Request) -> dict[str, str]:
    headers = {k.lower(): v for k, v in request.headers.items()}
    client_ip = _extract_remote_ip(request)
    user_agent = headers.get("user-agent", "")
    explicit_client_key = headers.get("x-client-key") or ""
    explicit_client_name = headers.get("x-client-name") or headers.get("x-client-id") or ""
    client_name = explicit_client_name.strip() or (user_agent[:80] if user_agent else "")
    client_kind = _classify_client_kind(client_ip, headers)
    key_suffix = explicit_client_key.strip().lower() or explicit_client_name.strip().lower()
    client_key = f"{client_ip}|{key_suffix}" if key_suffix else client_ip
    client_label = " | ".join(p for p in [client_name, client_ip] if p) or client_key
    return {
        "client_key": client_key,
        "client_label": client_label[:140],
        "client_ip": client_ip,
        "client_ip_source": "forwarded-header" if headers.get("x-forwarded-for") else "socket-peer",
        "client_kind": client_kind,
        "client_details": f"ua={user_agent[:120]} | host={headers.get('host', '')}",
        "client_priority_header": headers.get("x-client-priority", ""),
    }


async def _record_request_outcome(
    request_id: str,
    method: str,
    path: str,
    client_info: dict[str, str],
    status_code: int | None,
    duration_ms: float,
    request_body: bytes,
    response_body: bytes,
    error_text: str = "",
) -> None:
    body_preview = request_body.decode("utf-8", errors="ignore")[:200].replace("\n", " ") if request_body else ""
    response_preview = response_body.decode("utf-8", errors="ignore")[:200].replace("\n", " ") if response_body else ""
    await state.add_history_entry(
        {
            "request_id": request_id,
            "method": method,
            "path": path,
            "client_key": client_info["client_key"],
            "client_label": client_info["client_label"],
            "client_ip": client_info["client_ip"],
            "client_ip_source": client_info["client_ip_source"],
            "client_kind": client_info["client_kind"],
            "client_details": client_info["client_details"],
            "client_priority": await state.get_client_priority(client_info["client_key"]),
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "enqueue_time": None,
            "completed_at": now_iso(),
            "question_preview": body_preview,
            "answer_preview": response_preview if not error_text else error_text,
            "request_preview": body_preview,
            "response_preview": response_preview,
            "request_details": body_preview,
            "response_details": response_preview if not error_text else error_text,
            "request_model": "",
            "request_prompt": body_preview,
            "response_model": "",
            "response_text": response_preview,
            "response_created_at": "",
            "response_done_reason": "",
            "response_done": "",
            "error": error_text,
        }
    )


def _extract_requested_model(path: str, request_body: bytes) -> str:
    """Read model name from known Ollama endpoints."""
    if path not in {"/api/generate", "/api/chat", "/api/embeddings", "/api/embed", "/api/show"}:
        return ""
    if not request_body:
        return ""
    try:
        payload = json.loads(request_body.decode("utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    # Some endpoints use "model", while /api/show typically uses "name".
    model = payload.get("model") or payload.get("name")
    return str(model).strip() if model is not None else ""


def _is_self_backend(target_url: str, request: Request) -> bool:
    """Detect backend configuration that points to this proxy instance."""
    try:
        parsed = urlparse(target_url)
        backend_host = (parsed.hostname or "").lower()
        backend_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        request_port = request.url.port or (443 if request.url.scheme == "https" else 80)
    except Exception:
        return False
    local_hosts = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
    return backend_host in local_hosts and backend_port == request_port


def _is_stream_request(path: str, body: bytes) -> bool:
    """Return True if this is a streaming Ollama request (the default for generate/chat)."""
    if path not in {"/api/generate", "/api/chat"}:
        return False
    if not body:
        return True  # Ollama defaults to stream=true when omitted
    try:
        payload = json.loads(body.decode("utf-8"))
        return payload.get("stream", True) is not False
    except Exception:
        return True  # unparseable body — assume streaming


async def _queue_worker() -> None:
    """Background worker: dequeue non-streaming requests and forward to Ollama one at a time."""
    while True:
        _, _, _, item = await state.queue.get()
        future: asyncio.Future = item["future"]
        request_id: str = item["request_id"]

        async with state.lock:
            state.pending.pop(request_id, None)
            state.active += 1

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(state.upstream_timeout, connect=30.0)
            ) as client:
                upstream = await client.request(
                    method=item["method"],
                    url=item["target_url"],
                    params=item["query_params"],
                    headers=item["headers"],
                    content=item["body"],
                )
            if not future.done():
                future.set_result(upstream)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
        finally:
            async with state.lock:
                state.active -= 1
            state.queue.task_done()


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker_tasks = [asyncio.create_task(_queue_worker()) for _ in range(state.workers)]
    yield
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)


app = FastAPI(title="SharedOllama Proxy", lifespan=lifespan)


@app.get("/monitor/api/state")
async def monitor_state(request: Request) -> Any:
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
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
            "backend_url": state.backend_url,
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
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
    if auth_error:
        return auth_error
    return {"clients": await state.client_snapshot()}


@app.post("/monitor/api/clients/{client_key}/state")
async def update_client_state(request: Request, client_key: str) -> Any:
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
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
    await state.add_log("WARN", "Client state changed", client_key=client_key, state=next_state)
    await state.record_metric_snapshot()
    return {"client": updated, "cancelled_queued": 0}


@app.post("/monitor/api/admin/config")
async def admin_config(request: Request) -> Any:
    """Receive a live config update from the admin process."""
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
    if auth_error:
        return auth_error
    payload = await request.json()
    saved = await state.update_runtime_config(
        backend_url=str(payload.get("backend_url", "")).strip(),
        shared_port=11434,
        ollama_host=str(payload.get("ollama_host", "")).strip(),
        ollama_port=payload.get("ollama_port", 11435),
    )
    return {"ok": True, "config": saved}


@app.post("/monitor/api/admin/refresh-models")
async def admin_refresh_models(request: Request) -> Any:
    """Refresh the proxy's model cache from the Ollama backend."""
    auth_error = await ensure_monitor_auth(request, state.monitor_token)
    if auth_error:
        return auth_error
    await state.refresh_models()
    return {"ok": True, "count": len(state.models_cache)}


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(full_path: str, request: Request) -> Response:
    """Proxy all requests to the configured Ollama backend."""
    path = f"/{full_path}"
    target_url = f"{state.backend_url}{path}"
    started = time.time()
    client_info = _build_client_info(request)
    await state.register_client(
        client_key=client_info["client_key"],
        client_label=client_info["client_label"],
        client_ip=client_info["client_ip"],
        client_ip_source=client_info["client_ip_source"],
        client_kind=client_info["client_kind"],
        client_details=client_info["client_details"],
    )

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
        await _record_request_outcome(
            request_id=str(uuid.uuid4()),
            method=request.method.upper(),
            path=path,
            client_info=client_info,
            status_code=429,
            duration_ms=(time.time() - started) * 1000,
            request_body=await request.body(),
            response_body=b"",
            error_text="rate limit exceeded",
        )
        await state.record_metric_snapshot()
        return JSONResponse({"error": "rate limit exceeded"}, status_code=429)

    if _is_self_backend(target_url, request):
        return Response(
            content=json.dumps({
                "error": "Invalid backend_url: points to proxy itself. Update backend in admin panel.",
                "backend_url": state.backend_url,
            }),
            status_code=500,
            media_type="application/json",
        )

    filtered_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    request_body = await request.body()

    request_id = str(uuid.uuid4())
    requested_model = _extract_requested_model(path, request_body)
    if requested_model:
        resolved_model = state.resolve_model_name(requested_model)
        if not state.model_allowed(resolved_model):
            return Response(
                content=json.dumps({"error": f"Model not allowed: {resolved_model}"}),
                status_code=403,
                media_type="application/json",
            )
        model_ready = await state.ensure_model_available(resolved_model, request_id=request_id)
        if not model_ready:
            return Response(
                content=json.dumps({"error": f"Failed to auto-load model: {resolved_model}"}),
                status_code=503,
                media_type="application/json",
            )

    if _is_stream_request(path, request_body):
        return await _forward_direct(
            request_id=request_id,
            method=request.method,
            target_url=target_url,
            query_params=request.query_params,
            headers=filtered_headers,
            body=request_body,
            path=path,
            client_info=client_info,
            started=started,
        )
    else:
        return await _forward_queued(
            request_id=request_id,
            method=request.method,
            target_url=target_url,
            query_params=request.query_params,
            headers=filtered_headers,
            body=request_body,
            path=path,
            client_info=client_info,
            started=started,
        )


async def _forward_direct(
    request_id: str,
    method: str,
    target_url: str,
    query_params: Any,
    headers: dict[str, str],
    body: bytes,
    path: str,
    client_info: dict[str, str],
    started: float,
) -> Response:
    """Forward a streaming request to Ollama with true HTTP streaming (no buffering)."""
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(state.upstream_timeout, connect=30.0))
        req = client.build_request(
            method=method,
            url=target_url,
            params=query_params,
            headers=headers,
            content=body,
        )
        upstream = await client.send(req, stream=True)

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-encoding", "transfer-encoding", "connection", "content-length"}
        }

        # Record at first-byte time; response body not available for streams
        await _record_request_outcome(
            request_id=request_id,
            method=method.upper(),
            path=path,
            client_info=client_info,
            status_code=upstream.status_code,
            duration_ms=(time.time() - started) * 1000,
            request_body=body,
            response_body=b"",
        )
        async with state.lock:
            state.processed_total += 1
        await state.record_metric_snapshot()

        async def generate():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            generate(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    except Exception as exc:
        async with state.lock:
            state.failed_total += 1
        await _record_request_outcome(
            request_id=request_id,
            method=method.upper(),
            path=path,
            client_info=client_info,
            status_code=502,
            duration_ms=(time.time() - started) * 1000,
            request_body=body,
            response_body=b"",
            error_text=str(exc),
        )
        await state.record_metric_snapshot()
        return Response(
            content=json.dumps({"error": f"Proxy error: {str(exc)}"}),
            status_code=502,
            media_type="application/json",
        )


async def _forward_queued(
    request_id: str,
    method: str,
    target_url: str,
    query_params: Any,
    headers: dict[str, str],
    body: bytes,
    path: str,
    client_info: dict[str, str],
    started: float,
) -> Response:
    """Enqueue a non-streaming request and wait for a worker to process it."""
    # x-client-priority header overrides the stored client priority for this request
    priority = await state.get_client_priority(client_info["client_key"])
    header_priority = parse_priority_value(client_info.get("client_priority_header", ""))
    if header_priority is not None:
        priority = header_priority

    enqueue_time = time.time()
    future: asyncio.Future[httpx.Response] = asyncio.get_event_loop().create_future()

    async with state.lock:
        state.queue_sequence += 1
        seq = state.queue_sequence

    body_preview = body.decode("utf-8", errors="ignore")[:200].replace("\n", " ")

    async with state.lock:
        state.pending[request_id] = QueueItemInfo(
            request_id=request_id,
            method=method.upper(),
            path=path,
            enqueue_time=enqueue_time,
            body_preview=body_preview,
            request_details=body_preview,
            request_model=_extract_requested_model(path, body),
            request_prompt=body_preview,
            client_key=client_info["client_key"],
            client_label=client_info["client_label"],
            client_ip=client_info["client_ip"],
            client_ip_source=client_info["client_ip_source"],
            client_kind=client_info["client_kind"],
            client_details=client_info["client_details"],
            client_priority=priority,
        )

    try:
        state.queue.put_nowait((priority, enqueue_time, seq, {
            "future": future,
            "request_id": request_id,
            "method": method,
            "target_url": target_url,
            "query_params": query_params,
            "headers": headers,
            "body": body,
        }))
    except asyncio.QueueFull:
        async with state.lock:
            state.pending.pop(request_id, None)
        await state.add_log("WARN", "Queue full, request rejected", path=path, client_key=client_info["client_key"])
        return JSONResponse({"error": "queue full, try again later"}, status_code=503)

    await state.add_log("INFO", "Request queued", request_id=request_id, priority=priority, path=path)
    await state.record_metric_snapshot()

    try:
        upstream_response = await asyncio.wait_for(asyncio.shield(future), timeout=state.upstream_timeout)
    except asyncio.TimeoutError:
        future.cancel()
        async with state.lock:
            state.pending.pop(request_id, None)
        await _record_request_outcome(
            request_id=request_id,
            method=method.upper(),
            path=path,
            client_info=client_info,
            status_code=504,
            duration_ms=(time.time() - started) * 1000,
            request_body=body,
            response_body=b"",
            error_text="queue timeout",
        )
        await state.record_metric_snapshot()
        return JSONResponse({"error": "queue timeout"}, status_code=504)
    except Exception as exc:
        async with state.lock:
            state.failed_total += 1
        await _record_request_outcome(
            request_id=request_id,
            method=method.upper(),
            path=path,
            client_info=client_info,
            status_code=502,
            duration_ms=(time.time() - started) * 1000,
            request_body=body,
            response_body=b"",
            error_text=str(exc),
        )
        await state.record_metric_snapshot()
        return Response(
            content=json.dumps({"error": f"Proxy error: {str(exc)}"}),
            status_code=502,
            media_type="application/json",
        )

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in {"content-encoding", "transfer-encoding", "connection", "content-length"}
    }

    await _record_request_outcome(
        request_id=request_id,
        method=method.upper(),
        path=path,
        client_info=client_info,
        status_code=upstream_response.status_code,
        duration_ms=(time.time() - started) * 1000,
        request_body=body,
        response_body=upstream_response.content,
    )
    async with state.lock:
        state.processed_total += 1
    await state.record_metric_snapshot()

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11434)
