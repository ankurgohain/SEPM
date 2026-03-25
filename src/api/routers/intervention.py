# for prod, prediction cache handled by Redis, locally a dict is used 

from __future__ import annotations
import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from src.api.model_registry import ModelRegistry, get_registry
from src.api.schemas import ( INTERVENTION_DESCRIPTIONS, InterventionDetail, InterventionType, PredictRequest)

logger = logging.getLogger("learnflow.routers.intervention")
router = APIRouter(prefix="/intervention", tags=["Intervention"])

_prediction_cache: dict[str, dict] = {}
_acknowledged:     set[str]        = set()
 
 
def _cache_key(learner_id: str) -> str:
    return learner_id.strip().lower()
 
def store_prediction(learner_id: str, prediction: dict) -> None:
    _prediction_cache[_cache_key(learner_id)] = {
        **prediction,
        "cached_at": time.time(),
    }
    # Remove stale acknowledgement when a new prediction arrives
    _acknowledged.discard(_cache_key(learner_id))

# GET /intervention/{learner_id}
@router.get("/{learner_id}",response_model=InterventionDetail, summary="Get active intervention for user",
            description="""Returns current intervention plan for leaner. **404** if no prediction made;**200** with 'intervention: CONTINUE_STANDARD_PATH' """)

async def get_intervention(learner_id: str, registry: ModelRegistry=Depends(get_registry))-> InterventionDetail:
    key = _cache_key(learner_id)
    cached = _prediction_cache.get(key)

    if not cached:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=("run POST /predict/learner first"),)
    
    intervention_str = cached.get("Intervention", "CONTINUE_STANDARD_PATH")
    dropout_tier = cached.get("dropout_tier", "low")
    dropout_risk = cached.get("dropout_risk", 0.0)
    mastery_prob = cached.get("mastery_prob", 0.0)
    intervention_type = InterventionType(intervention_str)
    description = INTERVENTION_DESCRIPTIONS.get(intervention_type, "No description available.")
    action_url = (
        f"/dashboard/learners/{learner_id}/interventions"
        if intervention_type != InterventionType.continue_standard
        else None
    )
 
    return InterventionDetail(
        learner_id   = learner_id,
        intervention = intervention_type,
        dropout_tier = dropout_tier,
        dropout_risk = round(dropout_risk, 4),
        mastery_prob = round(mastery_prob, 4),
        description  = description,
        action_url   = action_url,
    )

#  POST /intervention/{learner_id}/acknowledge
@router.post(
    "/{learner_id}/acknowledge",
    summary="Acknowledge an intervention",
    description="Mark the current intervention for a learner as seen by an instructor.",
    status_code=status.HTTP_200_OK,
)

async def acknowledge_intervention(learner_id: str) -> JSONResponse:
    key = _cache_key(learner_id)
    if key not in _prediction_cache:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = f"No intervention found for learner {learner_id}",)
    _acknowledged.add(key)
    logger.info(f"Intervention acknowledged for learner {learner_id}")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "learner_id":   learner_id,
            "acknowledged": True,
            "message":      "Intervention marked as acknowledged.",
        },
    )

#  all active intervetions
@router.get(
    "/",
    summary="List all active interventions",
    description="Returns all learners with a non-standard, unacknowledged intervention.",
    status_code=status.HTTP_200_OK,
)
async def list_interventions() -> JSONResponse:
    active = []
    for key, cached in _prediction_cache.items():
        if key in _acknowledged:
            continue
        iv = cached.get("intervention", "CONTINUE_STANDARD_PATH")
        if iv == "CONTINUE_STANDARD_PATH":
            continue
        active.append({
            "learner_id":    key,
            "intervention":  iv,
            "dropout_tier":  cached.get("dropout_tier", "low"),
            "dropout_risk":  round(cached.get("dropout_risk", 0.0), 4),
            "mastery_prob":  round(cached.get("mastery_prob", 0.0), 4),
            "cached_at":     cached.get("cached_at"),
        })
 
    # Sort by dropout_risk descending
    active.sort(key=lambda x: x["dropout_risk"], reverse=True)
 
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"count": len(active), "interventions": active},
    )