from __future__ import annotations
from enum import Enum
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

class ModuleID(int, Enum):
    python_basics = 0
    data_structures = 1
    ml_fundamentals = 2
    deep_learning=3
    nlp_basics = 4
    reinforcement_learning = 5

class DropoutTier(str, Enum):
    high = "high"
    medium="medium"
    low  = "low"

class InterventionType(str, Enum):
    alert_instructor = "ALERT_INSTRUCTOR"
    assign_remedial = "ASSIGN_REMEDIAL_MODULE"
    motivational_nudge = "SEND_MOTIVATIONAL_NUDGE"
    award_badge = "AWARD_BADGE"
    continue_standard = "CONTINUE_STANDARD_PATH"

class LearnerSession(BaseModel):
    quiz_score: float = Field(..., ge=0, le=100, description="Quiz score for the session (0-100)")
    engagement_rate: float = Field(..., ge = 0.0, le=1.0, description="Engagement rate for the session (0.0-1.0)")
    hint_count: int = Field(..., ge=0, le=10, description="Number of hints used in the session (0-10)")
    session_duration: float = Field(..., ge=0.0, le=120.0, description="Duration of the session in minutes (0-120)")
    correct_attempts: int = Field(..., ge=0, le=50, description="Number of correct attempts in the session (0-50)")
    incorrect_attempts: int = Field(..., ge=0, le=50, description="Number of incorrect attempts in the session (0-50)")
    module_id: ModuleID = Field(..., description="Module ID for the session")

    model_config = {"use_enum_values": True}
    

class PredictRequest(BaseModel):
    """
    POST /predict/learner
 
    Accepts a sequence of 1–10 learning sessions for a single learner
    and returns LSTM-predicted performance, mastery probability, and
    dropout risk with recommended intervention.
    """
 
    learner_id: str = Field(..., min_length=1, max_length=64,
                            description="Unique learner identifier")
    sessions:   list[LearnerSession] = Field(
        ..., min_length=1, max_length=10,
        description="Ordered sequence of learning sessions (oldest → newest)"
    )
 
    @field_validator("sessions")
    @classmethod
    def sessions_not_empty(cls, v):
        if not v:
            raise ValueError("sessions must contain at least one entry")
        return v
 
 
class AttentionWeights(BaseModel):
    """Per-timestep attention weights from the Bahdanau attention layer."""
    timesteps: list[int]   = Field(..., description="Timestep indices")
    weights:   list[float] = Field(..., description="Normalised attention weights (sum ≈ 1)")
 
 
class PredictResponse(BaseModel):
    """
    Prediction result for a single learner.
 
    All probability values are in [0, 1].
    performance_score is in [0, 100].
    """
 
    learner_id: str= Field(..., description="Echoed from request")
    performance_score: float= Field(..., ge=0, le=100, description="Predicted next-session score")
    mastery_prob:float= Field(..., ge=0, le=1, description="Probability of module mastery")
    dropout_risk:float= Field(..., ge=0, le=1, description="Probability of dropout in next 3 sessions")
    dropout_tier: DropoutTier
    intervention: InterventionType
    attention_weights: Optional[AttentionWeights] = Field( None, description="Attention weights if explain=true was passed")
    model_version: str= Field(default="1.0.0")
    
class BatchPredictRequest(BaseModel):
    """POST /predict/batch — up to 100 learners in one call."""
 
    learners: list[PredictRequest] = Field(
        ..., min_length=1, max_length=100,
        description="List of learner sequences to score"
    )
 
    @model_validator(mode="after")
    def unique_learner_ids(self):
        ids = [l.learner_id for l in self.learners]
        if len(ids) != len(set(ids)):
            raise ValueError("All learner_id values in a batch must be unique")
        return self
 
 
class BatchPredictResponse(BaseModel):
    results:     list[PredictResponse]
    total:       int
    failed:      int = 0
    errors:      list[str] = Field(default_factory=list)

# "GET /intervention/{learner_id}
class InterventionDetail(BaseModel):
 
    learner_id:     str
    intervention:   InterventionType
    dropout_tier:   DropoutTier
    dropout_risk:   float = Field(..., ge=0, le=1)
    mastery_prob:   float = Field(..., ge=0, le=1)
    description:    str
    action_url:     Optional[str] = None
 
 
INTERVENTION_DESCRIPTIONS: dict[str, str] = {
    InterventionType.alert_instructor:   "Learner shows critical dropout signals. Immediate instructor follow-up recommended.",
    InterventionType.assign_remedial:    "Learner struggles with core concepts. Remedial content has been queued.",
    InterventionType.motivational_nudge: "Moderate dropout risk detected. A motivational prompt has been sent.",
    InterventionType.award_badge:        "Learner demonstrates high mastery. Badge awarded and next module unlocked.",
    InterventionType.continue_standard:  "Learner is on track. No intervention required at this time.",
}
class HealthResponse(BaseModel):
    status:        str   = "ok"
    model_loaded:  bool  = True
    model_version: str   = "1.0.0"
    uptime_seconds: float = 0.0
 
class WSEventType(str, Enum):
    prediction  = "prediction"
    intervention = "intervention"
    error       = "error"
    ping        = "ping"
    pong        = "pong"
 
 
class WSIncomingMessage(BaseModel):
    """Message sent by the client over the WebSocket."""
    event:      WSEventType
    learner_id: Optional[str]              = None
    sessions:   Optional[list[LearnerSession]] = None
 
 
class WSOutgoingMessage(BaseModel):
    """Message broadcast to the client over the WebSocket."""
    event:   WSEventType
    payload: dict