"""
predictor.py
============
Single-learner real-time inference for the LearnFlow LSTM model.

Public API
----------
    from src.inference.predictor import LearnerPredictor, PredictionResult

    predictor = LearnerPredictor.from_registry(registry)
    result    = predictor.predict(num_seq, cat_seq)

    # With attention weights (for explainability / dashboards):
    result    = predictor.predict(num_seq, cat_seq, return_attention=True)
    print(result.attention_weights)   # np.ndarray shape (seq_len,)

Design notes
------------
• `PredictionResult` is a frozen dataclass — callers can't accidentally
  mutate prediction state.
• `LearnerPredictor` holds *no* mutable state after construction; it is
  safe to share across async worker threads.
• Attention extraction uses a lazily-built sub-model that is cached on
  the first call so there is no per-request graph overhead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import Model

logger = logging.getLogger("learnflow.inference.predictor")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Tier thresholds — dropout_risk value that places a learner in each tier.
# Evaluated in descending order; the first match wins.
_TIER_THRESHOLDS: list[tuple[str, float]] = [
    ("high",   0.65),
    ("medium", 0.40),
    ("low",    0.0),
]

# Intervention logic — evaluated top-to-bottom, first match wins.
# Each rule is (label, callable(mastery, dropout, perf) → bool).
_INTERVENTION_RULES: list[tuple[str, object]] = [
    ("ALERT_INSTRUCTOR",        lambda m, d, p: d > 0.65),
    ("ASSIGN_REMEDIAL_MODULE",  lambda m, d, p: m < 0.4 and p < 60),
    ("SEND_MOTIVATIONAL_NUDGE", lambda m, d, p: d > 0.40),
    ("AWARD_BADGE_AND_ADVANCE", lambda m, d, p: m > 0.75),
    ("CONTINUE_STANDARD_PATH",  lambda m, d, p: True),   # default
]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PredictionResult:
    """
    Immutable result object returned by :meth:`LearnerPredictor.predict`.

    Attributes
    ----------
    learner_id : str | None
        Optional caller-supplied identifier for correlation.
    performance_score : float
        Predicted quiz/session performance on a 0–100 scale.
    mastery_prob : float
        Probability [0, 1] that the learner has achieved module mastery.
    dropout_risk : float
        Probability [0, 1] of learner dropout within the next few sessions.
    dropout_tier : str
        Bucketed risk tier: ``"low"`` | ``"medium"`` | ``"high"``.
    intervention : str
        Recommended adaptive action — one of:
        ``ALERT_INSTRUCTOR``, ``ASSIGN_REMEDIAL_MODULE``,
        ``SEND_MOTIVATIONAL_NUDGE``, ``AWARD_BADGE_AND_ADVANCE``,
        ``CONTINUE_STANDARD_PATH``.
    attention_weights : np.ndarray | None
        Per-timestep attention weights shape ``(seq_len,)``, or ``None``
        when *return_attention* was ``False``.
    latency_ms : float
        Wall-clock inference time in milliseconds (excludes pre-processing).
    """

    learner_id:        Optional[str]
    performance_score: float
    mastery_prob:      float
    dropout_risk:      float
    dropout_tier:      str
    intervention:      str
    attention_weights: Optional[np.ndarray]
    latency_ms:        float

    # ── Convenience helpers ───────────────────────────────────────────────────

    def to_dict(self, *, include_attention: bool = False) -> dict:
        """Serialisable dictionary suitable for JSON responses."""
        d = {
            "learner_id":        self.learner_id,
            "performance_score": self.performance_score,
            "mastery_prob":      self.mastery_prob,
            "dropout_risk":      self.dropout_risk,
            "dropout_tier":      self.dropout_tier,
            "intervention":      self.intervention,
            "latency_ms":        round(self.latency_ms, 3),
        }
        if include_attention and self.attention_weights is not None:
            d["attention_weights"] = self.attention_weights.tolist()
        return d

    @property
    def is_at_risk(self) -> bool:
        """True when dropout tier is *medium* or *high*."""
        return self.dropout_tier in ("medium", "high")

    @property
    def has_mastered(self) -> bool:
        """True when mastery probability exceeds 50 %."""
        return self.mastery_prob > 0.50

    def __repr__(self) -> str:
        return (
            f"PredictionResult("
            f"learner_id={self.learner_id!r}, "
            f"perf={self.performance_score:.1f}, "
            f"mastery={self.mastery_prob:.3f}, "
            f"dropout={self.dropout_risk:.3f} [{self.dropout_tier}], "
            f"intervention={self.intervention!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class LearnerPredictor:
    """
    Real-time, single-learner inference wrapper around the trained LSTM.

    Parameters
    ----------
    model : keras.Model
        Compiled, trained LearnerProgressionLSTM (or compatible) model.
    scaler : MinMaxScaler
        The *fitted* scaler used during training.  Applied to raw
        numerical features before inference.
    seq_len : int
        Expected sequence length (number of timesteps).  Inputs shorter
        than this are left-padded with zeros; longer inputs are truncated
        to the most-recent *seq_len* timesteps.
    num_features : int
        Number of numerical feature columns per timestep (default 6).

    Usage
    -----
    ::

        predictor = LearnerPredictor(model, scaler)
        result = predictor.predict(num_seq, cat_seq, learner_id="L-001")

        # With attention:
        result = predictor.predict(num_seq, cat_seq, return_attention=True)

    Thread safety
    -------------
    The instance holds no mutable state after ``__init__`` completes.
    Multiple async workers may call :meth:`predict` concurrently.
    TensorFlow's GIL-like graph execution serialises the actual GPU/CPU
    kernel work, so there is no data-race risk.
    """

    def __init__(
        self,
        model:        Model,
        scaler:       MinMaxScaler,
        seq_len:      int = 10,
        num_features: int = 6,
    ) -> None:
        self._model        = model
        self._scaler       = scaler
        self._seq_len      = seq_len
        self._num_features = num_features

        # Lazily-built explainability sub-model (cached after first call).
        self._attention_model: Optional[Model] = None

        logger.info(
            "LearnerPredictor ready — seq_len=%d  num_features=%d",
            seq_len, num_features,
        )

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_registry(cls, registry, **kwargs) -> "LearnerPredictor":
        """
        Convenience constructor that pulls the model and scaler from a
        :class:`src.api.model_registry.ModelRegistry` instance.

        ::

            from src.api.model_registry import get_registry
            predictor = LearnerPredictor.from_registry(get_registry())
        """
        return cls(
            model=registry.model,
            scaler=registry.scaler,
            **kwargs,
        )

    # ── Core inference ────────────────────────────────────────────────────────

    def predict(
        self,
        num_seq:          np.ndarray,
        cat_seq:          np.ndarray,
        learner_id:       Optional[str] = None,
        return_attention: bool = False,
    ) -> PredictionResult:
        """
        Score a single learner.

        Parameters
        ----------
        num_seq : np.ndarray, shape ``(T, num_features)``
            Raw (unscaled) numerical feature matrix.  Rows are timesteps,
            columns are ``[quiz_score, engagement_rate, hint_count,
            session_duration, correct_attempts, incorrect_attempts]``.
        cat_seq : np.ndarray, shape ``(T,)``
            Integer-encoded module IDs corresponding to each timestep.
        learner_id : str, optional
            Caller-supplied identifier echoed back in the result.
        return_attention : bool
            When ``True``, populate :attr:`PredictionResult.attention_weights`
            with per-timestep Bahdanau attention scores (shape ``(seq_len,)``).

        Returns
        -------
        PredictionResult
        """
        num_seq = np.asarray(num_seq, dtype=np.float32)
        cat_seq = np.asarray(cat_seq, dtype=np.int32)

        X_num, X_cat = self._preprocess(num_seq, cat_seq)

        t0 = time.perf_counter()

        if return_attention:
            preds, attn = self._predict_with_attention(X_num, X_cat)
        else:
            preds = self._model.predict([X_num, X_cat], verbose=0)
            attn  = None

        latency_ms = (time.perf_counter() - t0) * 1_000

        perf    = float(preds["performance_score"][0, 0])
        mastery = float(preds["mastery_prob"][0, 0])
        dropout = float(preds["dropout_risk"][0, 0])

        tier         = _resolve_tier(dropout)
        intervention = _resolve_intervention(mastery, dropout, perf)

        logger.debug(
            "predict learner=%s perf=%.1f mastery=%.3f dropout=%.3f tier=%s "
            "intervention=%s latency=%.1f ms",
            learner_id, perf, mastery, dropout, tier, intervention, latency_ms,
        )

        return PredictionResult(
            learner_id        = learner_id,
            performance_score = round(perf, 2),
            mastery_prob      = round(mastery, 4),
            dropout_risk      = round(dropout, 4),
            dropout_tier      = tier,
            intervention      = intervention,
            attention_weights = attn,
            latency_ms        = latency_ms,
        )

    # ── Pre-processing ────────────────────────────────────────────────────────

    def _preprocess(
        self,
        num_seq: np.ndarray,
        cat_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate, pad/truncate, scale, and add the batch dimension.

        Returns
        -------
        X_num : (1, seq_len, num_features)  float32
        X_cat : (1, seq_len)                int32
        """
        T = num_seq.shape[0]
        S = self._seq_len

        # ── Validate feature count ────────────────────────────────────────────
        if num_seq.ndim != 2 or num_seq.shape[1] != self._num_features:
            raise ValueError(
                f"num_seq must be shape (T, {self._num_features}); "
                f"got {num_seq.shape}"
            )
        if cat_seq.ndim != 1 or len(cat_seq) != T:
            raise ValueError(
                f"cat_seq must be shape (T,) matching num_seq rows; "
                f"got {cat_seq.shape}"
            )

        # ── Pad or truncate ───────────────────────────────────────────────────
        if T < S:
            pad_rows = S - T
            num_seq  = np.vstack([np.zeros((pad_rows, self._num_features), dtype=np.float32), num_seq])
            cat_seq  = np.concatenate([np.zeros(pad_rows, dtype=np.int32), cat_seq])
        elif T > S:
            logger.warning("Input length %d > seq_len %d — truncating to most-recent %d timesteps.", T, S, S)
            num_seq = num_seq[-S:]
            cat_seq = cat_seq[-S:]

        # ── Scale numerical features ──────────────────────────────────────────
        scaled = self._scaler.transform(num_seq)          # (S, F)

        X_num = scaled[np.newaxis].astype(np.float32)     # (1, S, F)
        X_cat = cat_seq[np.newaxis].astype(np.int32)      # (1, S)
        return X_num, X_cat

    # ── Attention extraction ──────────────────────────────────────────────────

    def _get_attention_model(self) -> Model:
        """Build and cache the explainability sub-model on first call."""
        if self._attention_model is None:
            from keras._tf_keras import Model as KerasModel

            try:
                attn_layer  = self._model.get_layer("attention")
                attn_output = attn_layer.output   # (context, weights)

                self._attention_model = KerasModel(
                    inputs  = self._model.inputs,
                    outputs = {
                        "performance_score": self._model.output["performance_score"],
                        "mastery_prob":      self._model.output["mastery_prob"],
                        "dropout_risk":      self._model.output["dropout_risk"],
                        "attention_weights": attn_output[1],   # (batch, T)
                    },
                    name = "LearnerProgressionLSTM_Explainable",
                )
                logger.info("Attention sub-model built and cached.")
            except (ValueError, AttributeError) as exc:
                logger.warning(
                    "Could not build attention sub-model (%s). "
                    "Falling back to standard prediction without attention.",
                    exc,
                )
                self._attention_model = None

        return self._attention_model

    def _predict_with_attention(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
    ) -> tuple[dict, Optional[np.ndarray]]:
        """
        Run the explainability sub-model and extract attention weights.

        Returns
        -------
        preds : dict  — same structure as ``model.predict`` output
        attn  : np.ndarray shape ``(seq_len,)`` or ``None`` on failure
        """
        attn_model = self._get_attention_model()

        if attn_model is None:
            preds = self._model.predict([X_num, X_cat], verbose=0)
            return preds, None

        full = attn_model.predict([X_num, X_cat], verbose=0)
        attn_weights = full.pop("attention_weights")   # (1, T)

        return full, attn_weights[0].astype(np.float32)   # (T,)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_tier(dropout: float) -> str:
    for tier, threshold in _TIER_THRESHOLDS:
        if dropout >= threshold:
            return tier
    return "low"


def _resolve_intervention(mastery: float, dropout: float, perf: float) -> str:
    for label, rule in _INTERVENTION_RULES:
        if rule(mastery, dropout, perf):
            return label
    return "CONTINUE_STANDARD_PATH"