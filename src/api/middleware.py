"""
middleware.py
=============
ASGI middleware stack for the LearnFlow API.

Layers (outermost → innermost)
──────────────────────────────
1. RequestLoggingMiddleware  — structured JSON log of every request/response
2. APIKeyMiddleware          — bearer-token authentication (skips /health)
3. RateLimitMiddleware       — per-client sliding-window rate limiter (in-memory)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger("learnflow.middleware")

# CONFIG (read from env at import time)
API_KEY = os.getenv("LEARNFLOW_API_KEY", "dev-insecure-key-change-me")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
RATE_LIMIT_WINDOW = 60.0                                     

# REQUEST LOGGING
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Emits one structured log line per request containing:
      request_id, method, path, status_code, duration_ms, client_ip.

    The request_id is also injected into the response headers as
    X-Request-ID for correlation in distributed traces.
    """

    SKIP_PATHS = {"/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        request_id  = str(uuid.uuid4())
        start       = time.perf_counter()
        client_ip   = self._client_ip(request)

        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            logger.error(
                json.dumps({
                    "request_id": request_id,
                    "method":     request.method,
                    "path":       request.url.path,
                    "status":     500,
                    "duration_ms": round(duration, 2),
                    "client_ip":  client_ip,
                    "error":      str(exc),
                })
            )
            raise

        duration = (time.perf_counter() - start) * 1000
        logger.info(
            json.dumps({
                "request_id":  request_id,
                "method":      request.method,
                "path":        request.url.path,
                "status":      response.status_code,
                "duration_ms": round(duration, 2),
                "client_ip":   client_ip,
            })
        )

        response.headers["X-Request-ID"] = request_id
        return response

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2. API KEY AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates the Authorization: Bearer <token> header on all routes
    except those in OPEN_PATHS.

    The key is compared using a constant-time HMAC comparison
    (via hashlib) to resist timing attacks.
    """

    OPEN_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/ws"}

    def __init__(self, app: ASGIApp, api_key: str = API_KEY):
        super().__init__(app)
        self._key_hash = hashlib.sha256(api_key.encode()).digest()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Allow WebSocket upgrades and open paths unconditionally
        if any(path.startswith(p) for p in self.OPEN_PATHS):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return self._unauthorised("Missing Bearer token")

        token      = auth_header[len("Bearer "):]
        token_hash = hashlib.sha256(token.encode()).digest()

        # Constant-time compare
        if not self._safe_compare(token_hash, self._key_hash):
            return self._unauthorised("Invalid API key")

        return await call_next(request)

    @staticmethod
    def _safe_compare(a: bytes, b: bytes) -> bool:
        if len(a) != len(b):
            return False
        diff = 0
        for x, y in zip(a, b):
            diff |= x ^ y
        return diff == 0

    @staticmethod
    def _unauthorised(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"detail": detail},
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. IN-MEMORY RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter keyed on client IP.

    Each IP is allowed RATE_LIMIT_RPM requests per 60-second window.
    When exceeded the response is 429 Too Many Requests with a
    Retry-After header indicating how many seconds to wait.

    NOTE: This is an in-process store — suitable for a single-replica
    deployment.  For multi-replica, replace with a Redis-backed
    implementation (e.g. slowapi + aioredis).
    """

    SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(
        self,
        app:     ASGIApp,
        limit:   int   = RATE_LIMIT_RPM,
        window:  float = RATE_LIMIT_WINDOW,
    ):
        super().__init__(app)
        self._limit  = limit
        self._window = window
        # client_ip → deque of request timestamps
        self._buckets: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        client_ip = self._client_ip(request)
        now       = time.monotonic()
        bucket    = self._buckets[client_ip]

        # Evict timestamps outside the sliding window
        while bucket and bucket[0] < now - self._window:
            bucket.popleft()

        if len(bucket) >= self._limit:
            oldest      = bucket[0]
            retry_after = int(self._window - (now - oldest)) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "detail":      "Rate limit exceeded",
                    "limit":       self._limit,
                    "window_secs": int(self._window),
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"]     = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self._limit - len(bucket)))
        return response

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "0.0.0.0"