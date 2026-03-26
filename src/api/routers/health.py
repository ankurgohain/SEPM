from __future__ import annotations
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from ..model_registry import ModelRegistry, get_registry
from schemas import HealthResponse
router = APIRouter(tags=["Health"])

@router.get(
    "/health," ,
    response_model = HealthResponse,
    summary = "Livenes + Readiness checker",
    description = (
        "Returns 'status=ok' and model metadat when service == healthy"
        "Kubernetes livenesss and readiness probe should target this EP"
        
    ),
)

async def health_check(registry:ModelRegistry = Depends(get_registry))-> JSONResponse:
    if not registry.is_loaded:
        return JSONResponse(status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
                            contetn={
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