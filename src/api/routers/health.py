from __future__ import annotations
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from src.api.model_registry import ModelRegistry, get_registry
from src.api.schemas import HealthResponse
router = APIRouter(tags=["health"])

@router.get(
    "/health",
    response_model = HealthResponse,
    summary = "Livenes + Readiness checker",
    description = (
        "Returns 'status=ok' and model metadata when service == healthy"
        "Kubernetes livenesss and readiness probe should target this EP"
        
    ),
)

async def health_check(registry:ModelRegistry = Depends(get_registry))-> JSONResponse:
    if not registry.is_loaded:
        return JSONResponse(status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={
                                "status": 'unavailable',
                                "model_loaded": "unloaded",
                                "uptime": 0.0,
                                "detail": "Model registry not initialised",

                            },)
    return JSONResponse(
        status_code = status.HTTP_200_OK, 
        content={
            "status": "ok", 
            "model_loaded": True,
            "model_version": registry.version,
            "uptime_seconds": round(registry.uptime, 1),
            "predict_count": registry.predict_count,
        },
    )