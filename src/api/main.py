from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Local imports must be package-relative so this module can be imported as `src.api.main`.
from src.api.middleware import APIKeyMiddleware, RateLimitMiddleware, RequestLoggingMiddleware
from src.api.model_registry import get_registry
from src.api.routers import health, intervention, predict
from src.api.websocket_handler import router as ws_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("learnflow.main")


@asynccontextmanager
async def lifespan(app: FastAPI)->AsyncGenerator:
    logger.info("="*52)
    logger.info("LearnFlow API Starting up...")
    logger.info("="*52)

    registry = get_registry()
    checkpoint_env = os.getenv("MODEL_CHECKPOINT")
    checkpoint_path = Path(checkpoint_env) if checkpoint_env else None

    try:
        registry.load(checkpoint_path=checkpoint_path)
        logger.info(f"Model loaded successfully. v{registry.version}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

    yield
    logger.info("System shutdown done")

app = FastAPI(
    title = "APIs for LSTM network implementation",
    description = "This API serves endpoints for predicting learner mastery and dropout risk based on their interaction data.",
    version = "1.0.0",
    docs_url="/docs",
    redoc_url ="/redoc",
    lifespan = lifespan,
)
origins = [
    "http://localhost:5500",
    "http://q27.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins = os.getenv("CORS_ORIGINS", "*").split(","),
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RateLimitMiddleware)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(intervention.router)
app.include_router(ws_router)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Interval Server error",
            "path": str(request.url.path),
            "request_id": getattr(request.state, "request_id", None),
        },
    )

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": "LEarnflow API", 
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    })

@app.get("/")
def root():
    return {"message": "API running with CORS"}
@app.get("/api/data")
def read_data():
    return {"data": "Frontend and backend connected"}