from __future__ import annotations
import logging
from typing import Annotated
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from ..model_registry import ModelRegistry, get_registry
from ..schemas import (AttentionWeights, BatchPredictRequest, BatchPredictResponse, InterventionType, PredictRequest, PredictResponse)

logger = logging.getLogger("learnflow.routes.predict")
router = APIRouter(prefix="/predict", tags=["predict"])
SEQ_LEN = 10
NUM_FEATURES=6
NUM_MODULES=6

def _sessions_to_arrays( request: PredictRequest) ->tuple[np.ndarray, np.ndarray]:
    sessions = request.sessions
    T = len(sessions)
    num_raw = np.zeros((SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    cat_raw = np.zeros((SEQ_LEN,), dtype=np.int32)
    offset = SEQ_LEN -T 
    for i,s in enumerate(sessions):
        idx = offset + i
        num_raw[idx] = [s.quiz_score, s.engagement_rate, s.hint_count, s.session_duration, s.correct_attempts, s.incorrect_attempts]
        cat_raw[idx] = int(s.module_id)
    return num_raw, cat_raw

def _build_response(learner_id: str, pred:dict, return_attention: bool) -> PredictResponse: 
    attn = None
    if return_attention and pred.get("attention_weights"):
        aw = pred["attention_weights"]
        attn = AttentionWeights(timesteps=aw["timesteps"], weights = aw["weights"],)

    return PredictResponse(learner_id = learner_id, performance_score = pred["performance_score"], mastery_prob = pred["mastery_prob"], dropout_risk = pred["dropout_risk"], dropout_tier = pred["dropout_tier"], intervention = InterventionType(pred["intervention"]),
                           attention_weights = attn, model_version = pred["model_version"],)
    
@router.post("/learner", response_model=PredictResponse, status_code = status.HTTP_200_OK, summary = "Predict progression for single learner", description="Accept seq of 10 session of one learner, return 3 prediction signals" \
"-Performance Score (0-100)" \
"-Mastery Probability " \
"-Dropout Risk (Low , Medium, High)" \
"- Intervention ")

async def predict_learner(
    body: PredictRequest, 
    explain: Annotated[bool, Query(description="Return attn wt")] = False, registry: ModelRegistry =Depends(get_registry), )-> PredictResponse:
    if not registry.is_loaded:
        raise HTTPException(status_code = status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not ready ...",)
    num_seq , cat_seq = _sessions_to_arrays(body)
    num_batch = num_seq[np.newaxis]
    cat_batch = cat_seq[np.newaxis]

    try:
        results = registry.predict(
            num_sequences = num_batch, 
            cat_sequences = cat_batch, 
            return_attention = explain,)
    except Exception as e:
        logger.exception("Prediction failed for learner %s", body.learner_id)
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Prediction error:{e}",)
    return _build_response(body.learner_id, results[0], explain)

@router.post("/batch", response_model=BatchPredictResponse, status_code = status.HTTP_200_OK, summary="Batch prediction fro multiple users", 
             description="Stores upto 100 learners in a request; each user processed individually",)
async def predict_batch( body: BatchPredictRequest, registry: ModelRegistry=Depends(get_registry),) -> BatchPredictResponse:
    if not registry.is_loaded:
        raise HTTPException(status_code = status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not ready ...",)
    results: list[PredictResponse] = []
    error: list[str] = []
    failed: int =0

    num_list, cat_list, ids =[], [], []
    for req in body.learners:
        num_seq, cat_seq = _sessions_to_arrays(req)
        num_list.append(num_seq)
        cat_list.append(cat_seq)
        ids.append(req.learner_id)

    num_batch = np.stack(num_list)
    cat_batch = np.stack(cat_list)

    try:
        preds = registry.predict(num_batch, cat_batch, return_attention=False)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch prediction error: {e}",)
    for learner_id, pred in zip(ids, preds):
        try:
            results.append(_build_response(learner_id, pred, False))
        except Exception as e:
            failed+=1
            error.append(f"{learner_id}: {e}")
            logger.warning(f"Response build fail for {learner_id} {e}")
    return BatchPredictResponse(results = results, total=len(body.learners), failed=failed, error = error,)