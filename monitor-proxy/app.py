import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

OLLAMA_INTERNAL_URL = os.getenv("OLLAMA_INTERNAL_URL", "http://ollama:11434").rstrip("/")
DB_PATH = os.getenv("MONITOR_DB_PATH", "/data/monitor.db")
RETENTION_DAYS = int(os.getenv("MONITOR_RETENTION_DAYS", "30"))
ERROR_TEXT_LIMIT = 2000

logger = logging.getLogger("monitor_proxy")
http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
retention_task: asyncio.Task | None = None


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_epoch INTEGER NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                remote_ip TEXT,
                remote_location TEXT,
                model TEXT,
                memory_used_bytes INTEGER,
                credits_used INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                response_time_ms INTEGER NOT NULL,
                status_code INTEGER,
                error TEXT,
                prompt TEXT,
                answer TEXT
            )
            """
        )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_ts_epoch ON request_logs(ts_epoch)")
        await db.commit()


async def cleanup_old_records() -> int:
    cutoff = int(time.time()) - (RETENTION_DAYS * 86400)
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("DELETE FROM request_logs WHERE ts_epoch < ?", (cutoff,))
        await db.commit()
        return cursor.rowcount or 0


async def retention_loop() -> None:
    while True:
        try:
            await cleanup_old_records()
        except Exception as exc:
            logger.exception("Retention cleanup failed: %s", exc)
        await asyncio.sleep(3600)


def _extract_remote_ip(request: Request) -> str | None:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


def _extract_remote_location(request: Request) -> str | None:
    for header in ("cf-ipcountry", "x-vercel-ip-country", "x-country-code"):
        value = request.headers.get(header)
        if value:
            return value
    return None


def _safe_json_loads(value: bytes) -> Any | None:
    if not value:
        return None
    try:
        return json.loads(value.decode("utf-8"))
    except Exception:
        return None


def _extract_prompt(request_json: Any, path: str) -> str | None:
    if not isinstance(request_json, dict):
        return None
    if "prompt" in request_json and isinstance(request_json.get("prompt"), str):
        return request_json["prompt"]
    if path.startswith("/api/chat") and isinstance(request_json.get("messages"), list):
        return json.dumps(request_json["messages"], ensure_ascii=False)
    if path.startswith("/api/embed") and request_json.get("input") is not None:
        return json.dumps(request_json["input"], ensure_ascii=False)
    return None


def _extract_from_ndjson(response_text: str) -> tuple[str | None, int | None, int | None, str | None]:
    answer_parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None

    for line in response_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if model is None and isinstance(obj.get("model"), str):
            model = obj["model"]
        if isinstance(obj.get("response"), str):
            answer_parts.append(obj["response"])
        message = obj.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            answer_parts.append(message["content"])
        if "prompt_eval_count" in obj and isinstance(obj["prompt_eval_count"], int):
            prompt_tokens = obj["prompt_eval_count"]
        if "eval_count" in obj and isinstance(obj["eval_count"], int):
            completion_tokens = obj["eval_count"]

    answer = "".join(answer_parts) if answer_parts else None
    return answer, prompt_tokens, completion_tokens, model


def _extract_response_fields(
    response_bytes: bytes, content_type: str
) -> tuple[str | None, int | None, int | None, str | None]:
    answer: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None

    if "application/x-ndjson" in content_type:
        return _extract_from_ndjson(response_bytes.decode("utf-8", errors="ignore"))

    response_json = _safe_json_loads(response_bytes)
    if not isinstance(response_json, dict):
        return None, None, None, None

    if isinstance(response_json.get("response"), str):
        answer = response_json["response"]
    message = response_json.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        answer = message["content"]
    if isinstance(response_json.get("prompt_eval_count"), int):
        prompt_tokens = response_json["prompt_eval_count"]
    if isinstance(response_json.get("eval_count"), int):
        completion_tokens = response_json["eval_count"]
    if isinstance(response_json.get("model"), str):
        model = response_json["model"]

    return answer, prompt_tokens, completion_tokens, model


def _extract_error_message(response_bytes: bytes, content_type: str, status_code: int) -> str | None:
    if status_code < 400:
        return None
    if "application/json" in content_type:
        response_json = _safe_json_loads(response_bytes)
        if isinstance(response_json, dict):
            for key in ("error", "detail", "message"):
                value = response_json.get(key)
                if isinstance(value, str) and value.strip():
                    return value[:ERROR_TEXT_LIMIT]
    return f"Upstream HTTP {status_code}"


async def _extract_memory_used_bytes(model: str | None) -> int | None:
    if not model:
        return None
    try:
        response = await http_client.get(f"{OLLAMA_INTERNAL_URL}/api/ps")
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    models = payload.get("models")
    if not isinstance(models, list):
        return None

    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        if name == model:
            if isinstance(item.get("size_vram"), int):
                return item["size_vram"]
            if isinstance(item.get("size"), int):
                return item["size"]
            return None
    return None


async def _insert_log(entry: dict[str, Any]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO request_logs (
                ts_epoch, method, path, remote_ip, remote_location, model, memory_used_bytes,
                credits_used, prompt_tokens, completion_tokens, response_time_ms, status_code,
                error, prompt, answer
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["ts_epoch"],
                entry["method"],
                entry["path"],
                entry.get("remote_ip"),
                entry.get("remote_location"),
                entry.get("model"),
                entry.get("memory_used_bytes"),
                entry.get("credits_used"),
                entry.get("prompt_tokens"),
                entry.get("completion_tokens"),
                entry["response_time_ms"],
                entry.get("status_code"),
                entry.get("error"),
                entry.get("prompt"),
                entry.get("answer"),
            ),
        )
        await db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global retention_task
    await init_db()
    await cleanup_old_records()
    retention_task = asyncio.create_task(retention_loop())
    try:
        yield
    finally:
        if retention_task:
            retention_task.cancel()
        await http_client.aclose()


app = FastAPI(title="SharedOllama Monitor Proxy", lifespan=lifespan)


@app.get("/monitor/summary")
async def monitor_summary() -> dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        total_requests, total_credits, avg_response_time = (0, 0, 0.0)
        errors = 0
        async with db.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(credits_used), 0),
                COALESCE(AVG(response_time_ms), 0),
                COALESCE(SUM(CASE WHEN error IS NOT NULL OR status_code >= 400 THEN 1 ELSE 0 END), 0)
            FROM request_logs
            """
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                total_requests, total_credits, avg_response_time, errors = row

        model_usage: dict[str, int] = {}
        async with db.execute(
            """
            SELECT model, COUNT(*)
            FROM request_logs
            WHERE model IS NOT NULL
            GROUP BY model
            ORDER BY COUNT(*) DESC
            """
        ) as cursor:
            async for model, count in cursor:
                model_usage[model] = count

    return {
        "total_requests": total_requests,
        "total_credits_used": total_credits,
        "average_response_time_ms": round(float(avg_response_time), 2),
        "errors": errors,
        "model_usage": model_usage,
        "retention_days": RETENTION_DAYS,
    }


@app.get("/monitor/requests")
async def monitor_requests(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT
                id, ts_epoch, method, path, remote_ip, remote_location, model,
                memory_used_bytes, credits_used, prompt_tokens, completion_tokens,
                response_time_ms, status_code, error, prompt, answer
            FROM request_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = [dict(row) async for row in cursor]
    return {"items": rows, "count": len(rows)}


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(full_path: str, request: Request) -> Response:
    path = f"/{full_path}"
    target_url = f"{OLLAMA_INTERNAL_URL}{path}"
    ts_epoch = int(time.time())
    started = time.perf_counter()
    request_body = await request.body()
    request_json = _safe_json_loads(request_body)
    model = request_json.get("model") if isinstance(request_json, dict) else None
    prompt = _extract_prompt(request_json, path)
    remote_ip = _extract_remote_ip(request)
    remote_location = _extract_remote_location(request)

    filtered_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    try:
        upstream_response = await http_client.request(
            method=request.method,
            url=target_url,
            params=request.query_params,
            headers=filtered_headers,
            content=request_body,
        )
        response_time_ms = int((time.perf_counter() - started) * 1000)
        content_type = upstream_response.headers.get("content-type", "")
        response_bytes = upstream_response.content
        answer, prompt_tokens, completion_tokens, response_model = _extract_response_fields(
            response_bytes, content_type
        )
        if isinstance(response_model, str) and not model:
            model = response_model
        memory_used_bytes = await _extract_memory_used_bytes(model if isinstance(model, str) else None)
        credits_used = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            credits_used = prompt_tokens + completion_tokens
        elif isinstance(prompt_tokens, int):
            credits_used = prompt_tokens
        elif isinstance(completion_tokens, int):
            credits_used = completion_tokens

        await _insert_log(
            {
                "ts_epoch": ts_epoch,
                "method": request.method,
                "path": path,
                "remote_ip": remote_ip,
                "remote_location": remote_location,
                "model": model if isinstance(model, str) else None,
                "memory_used_bytes": memory_used_bytes,
                "credits_used": credits_used,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_time_ms": response_time_ms,
                "status_code": upstream_response.status_code,
                "error": _extract_error_message(
                    response_bytes, content_type, upstream_response.status_code
                ),
                "prompt": prompt,
                "answer": answer,
            }
        )

        response_headers = {
            key: value
            for key, value in upstream_response.headers.items()
            if key.lower() not in {"content-encoding", "transfer-encoding", "connection", "content-length"}
        }
        return Response(
            content=response_bytes,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type"),
        )
    except Exception as exc:
        response_time_ms = int((time.perf_counter() - started) * 1000)
        await _insert_log(
            {
                "ts_epoch": ts_epoch,
                "method": request.method,
                "path": path,
                "remote_ip": remote_ip,
                "remote_location": remote_location,
                "model": model if isinstance(model, str) else None,
                "memory_used_bytes": None,
                "credits_used": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "response_time_ms": response_time_ms,
                "status_code": 502,
                "error": str(exc)[:ERROR_TEXT_LIMIT],
                "prompt": prompt,
                "answer": None,
            }
        )
        raise HTTPException(status_code=502, detail=f"Proxy error: {exc}")
