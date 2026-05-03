"""Pure Ollama proxy for SharedOllama. Runs on standard port (11434)."""

import json
import uuid
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from .shared import MonitorState

state = MonitorState()


app = FastAPI(title="SharedOllama Proxy")


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


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(full_path: str, request: Request) -> Response:
    """Proxy all requests to the configured Ollama backend."""
    path = f"/{full_path}"
    target_url = f"{state.backend_url}{path}"

    if _is_self_backend(target_url, request):
        return Response(
            content=json.dumps(
                {
                    "error": "Invalid backend_url: points to proxy itself. Update backend in admin panel.",
                    "backend_url": state.backend_url,
                }
            ),
            status_code=500,
            media_type="application/json",
        )

    # Extract headers, excluding ones that should be regenerated
    filtered_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    # Get request body
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

    try:
        # Forward request to backend
        async with httpx.AsyncClient(timeout=httpx.Timeout(state.upstream_timeout, connect=30.0)) as client:
            upstream_response = await client.request(
                method=request.method,
                url=target_url,
                params=request.query_params,
                headers=filtered_headers,
                content=request_body,
            )

        # Prepare response
        response_headers = {
            key: value
            for key, value in upstream_response.headers.items()
            if key.lower() not in {"content-encoding", "transfer-encoding", "connection", "content-length"}
        }

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type"),
        )

    except Exception as exc:
        return Response(
            content=json.dumps({"error": f"Proxy error: {str(exc)}"}),
            status_code=502,
            media_type="application/json",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11434)
