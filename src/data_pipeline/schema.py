"""
src/data_pipeline/schema.py
===========================
Canonical data models for raw learner events flowing through the pipeline.

Events arrive from three sources:
  • Kafka topic (real-time LMS stream)
  • CSV / Parquet files (batch offline ingestion)
  • REST POST (direct API push from LMS webhooks)

All sources normalise to LearnerEvent before sequencing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


MODULES = [
    "python_basics",
    "data_structures",
    "ml_fundamentals",
    "deep_learning",
    "nlp_basics",
    "reinforcement_learning",
]

MODULE_TO_ID: dict[str, int] = {m: i for i, m in enumerate(MODULES)}


class LearnerEvent(BaseModel):
    """
    One learning session row — the atomic unit of the pipeline.

    Fields
    ------
    event_id        : UUID assigned at ingestion (auto-generated if absent)
    learner_id      : opaque learner identifier (string)
    timestamp       : when the session ended (ISO-8601 or Unix epoch float)
    module          : course module slug
    module_id       : integer encoding of module (derived if absent)
    quiz_score      : 0–100 raw quiz score for the session
    engagement_rate : 0–1 fraction of session time actively on-task
    hint_count      : number of hints requested (0–10)
    session_duration: session length in minutes
    correct_attempts: number of correct problem attempts
    incorrect_attempts: number of incorrect problem attempts
    badge_earned    : whether a badge was awarded this session
    """

    event_id:           str   = Field(default_factory=lambda: str(uuid.uuid4()))
    learner_id:         str   = Field(..., min_length=1, max_length=128)
    timestamp:          datetime
    module:             str   = Field(..., description="Module slug")
    module_id:          Optional[int] = Field(None, ge=0)

    quiz_score:         float = Field(..., ge=0.0,  le=100.0)
    engagement_rate:    float = Field(..., ge=0.0,  le=1.0)
    hint_count:         int   = Field(..., ge=0,    le=10)
    session_duration:   float = Field(..., ge=1.0,  le=240.0)
    correct_attempts:   int   = Field(..., ge=0,    le=50)
    incorrect_attempts: int   = Field(..., ge=0,    le=50)
    badge_earned:       bool  = Field(default=False)

    model_config = {"use_enum_values": True}

    @field_validator("module")
    @classmethod
    def validate_module(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if v not in MODULE_TO_ID:
            raise ValueError(
                f"Unknown module '{v}'. Valid modules: {MODULES}"
            )
        return v

    @model_validator(mode="after")
    def fill_module_id(self) -> "LearnerEvent":
        if self.module_id is None:
            object.__setattr__(self, "module_id", MODULE_TO_ID[self.module])
        return self

    def to_feature_row(self) -> dict:
        """Return the 6-element numerical feature vector + module_id."""
        return {
            "quiz_score":          self.quiz_score,
            "engagement_rate":     self.engagement_rate,
            "hint_count":          float(self.hint_count),
            "session_duration":    self.session_duration,
            "correct_attempts":    float(self.correct_attempts),
            "incorrect_attempts":  float(self.incorrect_attempts),
            "module_id":           self.module_id,
        }


class LearnerEventBatch(BaseModel):
    """A validated batch of events from one ingestion chunk."""

    events: list[LearnerEvent]
    source: str = "unknown"
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def learner_ids(self) -> list[str]:
        return list({e.learner_id for e in self.events})

    @property
    def size(self) -> int:
        return len(self.events)