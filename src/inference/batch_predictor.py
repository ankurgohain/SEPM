"""
batch_predictor.py
==================
Vectorised mini-batch inference and cohort-level segment reporting
for the LearnFlow LSTM model.

Public API
----------
    from src.inference.batch_predictor import BatchPredictor, BatchReport

    predictor = BatchPredictor(model, scaler)

    # Score a list of learners (each is a (num_seq, cat_seq) tuple):
    results = predictor.predict_batch(learner_inputs, learner_ids=ids)

    # Full segment report — tier breakdown, intervention counts, averages:
    report = predictor.segment_report(results)
    print(report.summary())

Design notes
------------
• Inputs are stacked into a single NumPy array per mini-batch so that
  TensorFlow dispatches *one* predict call per chunk, minimising Python
  overhead and allowing GPU batching.
• Sequences are normalised to ``seq_len`` via the same pad/truncate logic
  used by LearnerPredictor, so the two classes can safely share a scaler.
• ``BatchReport`` is a dataclass with both machine-readable dicts and a
  human-readable ``summary()`` string — convenient for logging and for
  serialising into API responses.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras import Model

from src.inference.predictor import (
    PredictionResult,
    _resolve_intervention,
    _resolve_tier,
)

logger = logging.getLogger("learnflow.inference.batch_predictor")

# Default number of learners to score in a single model.predict() call.
_DEFAULT_CHUNK_SIZE = 128


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TierBreakdown:
    """Counts and percentages for each dropout-risk tier."""

    low:    int = 0
    medium: int = 0
    high:   int = 0
    total:  int = 0

    @property
    def pct_low(self)    -> float: return self._pct(self.low)
    @property
    def pct_medium(self) -> float: return self._pct(self.medium)
    @property
    def pct_high(self)   -> float: return self._pct(self.high)

    def _pct(self, n: int) -> float:
        return round(n / self.total * 100, 2) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "low":        {"count": self.low,    "pct": self.pct_low},
            "medium":     {"count": self.medium, "pct": self.pct_medium},
            "high":       {"count": self.high,   "pct": self.pct_high},
            "total":      self.total,
        }


@dataclass
class CohortAverages:
    """Mean predicted values across the scored cohort."""

    performance_score: float
    mastery_prob:      float
    dropout_risk:      float

    def to_dict(self) -> dict:
        return {
            "mean_performance_score": round(self.performance_score, 3),
            "mean_mastery_prob":      round(self.mastery_prob,      4),
            "mean_dropout_risk":      round(self.dropout_risk,       4),
        }


@dataclass
class BatchReport:
    """
    Aggregate segment report produced by :meth:`BatchPredictor.segment_report`.

    Attributes
    ----------
    n_learners : int
        Total number of learners scored.
    tier_breakdown : TierBreakdown
        Per-tier counts and percentages for dropout risk.
    intervention_counts : dict[str, int]
        How many learners received each recommended intervention.
    cohort_averages : CohortAverages
        Mean performance, mastery, and dropout scores across the cohort.
    mastery_rate : float
        Fraction of learners with ``mastery_prob > 0.5``.
    at_risk_ids : list[str]
        Learner IDs (or index strings) of all *high*-tier learners.
    total_latency_ms : float
        Wall-clock time for the entire batch scoring run.
    throughput_lps : float
        Learners scored per second.
    """

    n_learners:          int
    tier_breakdown:      TierBreakdown
    intervention_counts: dict[str, int]
    cohort_averages:     CohortAverages
    mastery_rate:        float
    at_risk_ids:         list[str]
    total_latency_ms:    float
    throughput_lps:      float

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "n_learners":          self.n_learners,
            "tier_breakdown":      self.tier_breakdown.to_dict(),
            "intervention_counts": self.intervention_counts,
            "cohort_averages":     self.cohort_averages.to_dict(),
            "mastery_rate":        round(self.mastery_rate * 100, 2),
            "at_risk_ids":         self.at_risk_ids,
            "total_latency_ms":    round(self.total_latency_ms, 2),
            "throughput_lps":      round(self.throughput_lps, 1),
        }

    # ── Human-readable summary ────────────────────────────────────────────────

    def summary(self) -> str:
        tb = self.tier_breakdown
        ca = self.cohort_averages
        ic = self.intervention_counts
        sep  = "─" * 56
        rows = [
            "═" * 56,
            "  LearnFlow · Batch Scoring Report",
            "═" * 56,
            f"  Cohort size       : {self.n_learners:,} learners",
            f"  Mastery rate      : {self.mastery_rate * 100:.1f} %",
            "",
            "  Dropout-Risk Tier Breakdown",
            f"  {sep}",
            f"  {'Tier':<10}  {'Count':>6}  {'%':>7}",
            f"  {'─'*10}  {'─'*6}  {'─'*7}",
            f"  {'Low':<10}  {tb.low:>6,}  {tb.pct_low:>6.1f}%",
            f"  {'Medium':<10}  {tb.medium:>6,}  {tb.pct_medium:>6.1f}%",
            f"  {'High':<10}  {tb.high:>6,}  {tb.pct_high:>6.1f}%",
            "",
            "  Intervention Counts",
            f"  {sep}",
        ]
        for intervention, count in sorted(ic.items(), key=lambda kv: -kv[1]):
            rows.append(f"  {intervention:<35}  {count:>5,}")
        rows += [
            "",
            "  Cohort Averages",
            f"  {sep}",
            f"  Mean performance score  : {ca.performance_score:.2f}",
            f"  Mean mastery prob       : {ca.mastery_prob:.4f}",
            f"  Mean dropout risk       : {ca.dropout_risk:.4f}",
            "",
            "  Performance",
            f"  {sep}",
            f"  Total latency   : {self.total_latency_ms:,.1f} ms",
            f"  Throughput      : {self.throughput_lps:,.1f} learners / sec",
            "═" * 56,
        ]
        if self.at_risk_ids:
            rows += [
                f"  ⚠  {len(self.at_risk_ids)} high-risk learner(s):",
                "  " + ", ".join(self.at_risk_ids[:10])
                + ("  …" if len(self.at_risk_ids) > 10 else ""),
                "═" * 56,
            ]
        return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class BatchPredictor:
    """
    Vectorised mini-batch scorer for the LearnFlow LSTM.

    Parameters
    ----------
    model : keras.Model
        The trained LearnerProgressionLSTM.
    scaler : MinMaxScaler
        The *fitted* scaler used during training.
    seq_len : int
        Expected number of timesteps per learner (default 10).
    num_features : int
        Number of numerical feature columns per timestep (default 6).
    chunk_size : int
        Number of learners to score in a single ``model.predict()`` call.
        Tune this to balance GPU utilisation against memory pressure.

    Usage
    -----
    ::

        predictor = BatchPredictor(model, scaler, chunk_size=256)

        # learner_inputs: list of (num_seq, cat_seq) tuples
        results = predictor.predict_batch(learner_inputs, learner_ids=ids)
        report  = predictor.segment_report(results)
        print(report.summary())
    """

    def __init__(
        self,
        model:        Model,
        scaler:       MinMaxScaler,
        seq_len:      int = 10,
        num_features: int = 6,
        chunk_size:   int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._model        = model
        self._scaler       = scaler
        self._seq_len      = seq_len
        self._num_features = num_features
        self._chunk_size   = chunk_size

        logger.info(
            "BatchPredictor ready — seq_len=%d  num_features=%d  chunk_size=%d",
            seq_len, num_features, chunk_size,
        )

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_registry(cls, registry, **kwargs) -> "BatchPredictor":
        """
        Convenience constructor from a
        :class:`src.api.model_registry.ModelRegistry`.

        ::

            predictor = BatchPredictor.from_registry(get_registry())
        """
        return cls(
            model=registry.model,
            scaler=registry.scaler,
            **kwargs,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_batch(
        self,
        learner_inputs: Sequence[tuple[np.ndarray, np.ndarray]],
        learner_ids:    Optional[Sequence[str]] = None,
    ) -> list[PredictionResult]:
        """
        Score an arbitrary-sized list of learners.

        Parameters
        ----------
        learner_inputs : sequence of ``(num_seq, cat_seq)`` tuples
            Each ``num_seq`` has shape ``(T, num_features)`` and each
            ``cat_seq`` has shape ``(T,)``.  ``T`` need not equal
            ``seq_len`` — shorter sequences are left-padded, longer ones
            are truncated to the most-recent *seq_len* timesteps.
        learner_ids : sequence of str, optional
            Identifiers echoed back in each :class:`PredictionResult`.
            Must have the same length as *learner_inputs* if supplied.

        Returns
        -------
        list[PredictionResult]
            In the same order as *learner_inputs*.
        """
        n = len(learner_inputs)
        if not n:
            return []

        if learner_ids is None:
            learner_ids = [str(i) for i in range(n)]
        if len(learner_ids) != n:
            raise ValueError(
                f"learner_ids length {len(learner_ids)} != learner_inputs length {n}"
            )

        logger.info("Batch inference — %d learners  chunk_size=%d", n, self._chunk_size)

        t_start  = time.perf_counter()
        results  = []

        # ── Pre-process all inputs into stacked arrays ─────────────────────────
        X_num_all, X_cat_all = self._preprocess_batch(learner_inputs)
        # X_num_all : (n, seq_len, num_features)  float32
        # X_cat_all : (n, seq_len)                int32

        # ── Mini-batch inference ───────────────────────────────────────────────
        perf_all    = np.empty(n, dtype=np.float32)
        mastery_all = np.empty(n, dtype=np.float32)
        dropout_all = np.empty(n, dtype=np.float32)

        for chunk_start in range(0, n, self._chunk_size):
            chunk_end = min(chunk_start + self._chunk_size, n)

            X_num_chunk = X_num_all[chunk_start:chunk_end]
            X_cat_chunk = X_cat_all[chunk_start:chunk_end]

            preds = self._model.predict([X_num_chunk, X_cat_chunk], verbose=0)

            perf_all   [chunk_start:chunk_end] = preds["performance_score"].squeeze()
            mastery_all[chunk_start:chunk_end] = preds["mastery_prob"].squeeze()
            dropout_all[chunk_start:chunk_end] = preds["dropout_risk"].squeeze()

            logger.debug(
                "  chunk %d–%d scored (%.1f%%)",
                chunk_start, chunk_end - 1, chunk_end / n * 100,
            )

        total_ms = (time.perf_counter() - t_start) * 1_000

        # ── Assemble PredictionResult objects ─────────────────────────────────
        per_learner_ms = total_ms / n
        for i, lid in enumerate(learner_ids):
            perf    = float(perf_all[i])
            mastery = float(mastery_all[i])
            dropout = float(dropout_all[i])

            results.append(PredictionResult(
                learner_id        = lid,
                performance_score = round(perf,    2),
                mastery_prob      = round(mastery,  4),
                dropout_risk      = round(dropout,  4),
                dropout_tier      = _resolve_tier(dropout),
                intervention      = _resolve_intervention(mastery, dropout, perf),
                attention_weights = None,
                latency_ms        = round(per_learner_ms, 3),
            ))

        logger.info(
            "Batch complete — %d learners  total=%.1f ms  throughput=%.0f lps",
            n, total_ms, n / (total_ms / 1_000),
        )
        return results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def segment_report(
        self,
        results: list[PredictionResult],
        *,
        high_risk_threshold: float = 0.65,
    ) -> BatchReport:
        """
        Aggregate a list of :class:`PredictionResult` into a
        :class:`BatchReport` containing tier breakdown, intervention
        counts, and cohort averages.

        Parameters
        ----------
        results : list[PredictionResult]
            Output of :meth:`predict_batch`.
        high_risk_threshold : float
            Dropout-risk threshold above which a learner is flagged in
            :attr:`BatchReport.at_risk_ids`.  Defaults to 0.65.

        Returns
        -------
        BatchReport
        """
        if not results:
            return self._empty_report()

        n = len(results)
        t_start = time.perf_counter()

        # ── Vectorise for fast aggregation ─────────────────────────────────────
        perfs     = np.array([r.performance_score for r in results], dtype=np.float64)
        masteries = np.array([r.mastery_prob      for r in results], dtype=np.float64)
        dropouts  = np.array([r.dropout_risk       for r in results], dtype=np.float64)

        tier_counter = Counter(r.dropout_tier  for r in results)
        iv_counter   = Counter(r.intervention  for r in results)

        tier_breakdown = TierBreakdown(
            low    = tier_counter.get("low",    0),
            medium = tier_counter.get("medium", 0),
            high   = tier_counter.get("high",   0),
            total  = n,
        )

        cohort_averages = CohortAverages(
            performance_score = float(np.mean(perfs)),
            mastery_prob      = float(np.mean(masteries)),
            dropout_risk      = float(np.mean(dropouts)),
        )

        mastery_rate = float(np.mean(masteries > 0.5))

        at_risk_ids = [
            r.learner_id or str(i)
            for i, r in enumerate(results)
            if r.dropout_risk >= high_risk_threshold
        ]

        total_ms = (time.perf_counter() - t_start) * 1_000

        # Estimate batch wall-clock from per-result latency stored in results
        batch_wall_ms = sum(r.latency_ms for r in results)
        throughput    = n / max(batch_wall_ms / 1_000, 1e-9)

        logger.info(
            "Segment report — %d learners  high_risk=%d  mastery_rate=%.1f%%",
            n, len(at_risk_ids), mastery_rate * 100,
        )

        return BatchReport(
            n_learners          = n,
            tier_breakdown      = tier_breakdown,
            intervention_counts = dict(iv_counter),
            cohort_averages     = cohort_averages,
            mastery_rate        = mastery_rate,
            at_risk_ids         = at_risk_ids,
            total_latency_ms    = batch_wall_ms,
            throughput_lps      = throughput,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess_batch(
        self,
        learner_inputs: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate, pad/truncate, and scale all inputs.

        Returns
        -------
        X_num : (n, seq_len, num_features)  float32
        X_cat : (n, seq_len)                int32
        """
        n   = len(learner_inputs)
        S   = self._seq_len
        F   = self._num_features

        X_num_raw = np.zeros((n, S, F), dtype=np.float32)
        X_cat_out = np.zeros((n, S),    dtype=np.int32)

        for i, (num_seq, cat_seq) in enumerate(learner_inputs):
            num_seq = np.asarray(num_seq, dtype=np.float32)
            cat_seq = np.asarray(cat_seq, dtype=np.int32)

            T = num_seq.shape[0]

            # Validate
            if num_seq.ndim != 2 or num_seq.shape[1] != F:
                raise ValueError(
                    f"learner_inputs[{i}]: num_seq must be (T, {F}); got {num_seq.shape}"
                )
            if cat_seq.ndim != 1 or len(cat_seq) != T:
                raise ValueError(
                    f"learner_inputs[{i}]: cat_seq must be (T,); got {cat_seq.shape}"
                )

            # Pad or truncate
            if T < S:
                pad = S - T
                num_seq = np.vstack([np.zeros((pad, F), dtype=np.float32), num_seq])
                cat_seq = np.concatenate([np.zeros(pad, dtype=np.int32), cat_seq])
            elif T > S:
                num_seq = num_seq[-S:]
                cat_seq = cat_seq[-S:]

            X_num_raw[i] = num_seq
            X_cat_out[i] = cat_seq

        # Scale all numerical data in one vectorised call
        flat         = X_num_raw.reshape(-1, F)                      # (n*S, F)
        flat_scaled  = self._scaler.transform(flat).astype(np.float32)
        X_num_scaled = flat_scaled.reshape(n, S, F)

        return X_num_scaled, X_cat_out

    @staticmethod
    def _empty_report() -> BatchReport:
        return BatchReport(
            n_learners          = 0,
            tier_breakdown      = TierBreakdown(),
            intervention_counts = {},
            cohort_averages     = CohortAverages(0.0, 0.0, 0.0),
            mastery_rate        = 0.0,
            at_risk_ids         = [],
            total_latency_ms    = 0.0,
            throughput_lps      = 0.0,
        )