"""
src/data_pipeline/feature_engineering.py
=========================================
Transforms cleaned event batches into model-ready feature tensors.

Pipeline
────────
  LearnerEventBatch
    → per-learner sorted timeline
    → derived features (rolling stats, engagement_index, score_trend …)
    → windowed sequences of length SEQ_LEN
    → (X_num, X_cat) numpy arrays + label vector y

Derived features added to each session row
──────────────────────────────────────────
  score_delta          – score change from previous session
  rolling_score_3      – rolling mean of last-3 quiz scores
  rolling_engagement_3 – rolling mean of last-3 engagement rates
  attempt_ratio        – correct / (correct + incorrect + ε)
  hint_rate            – hint_count / max_hints (normalised)
  cumulative_badges    – running total of badges earned
  session_number       – position in learner's history (1-indexed)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_pipeline.schema import LearnerEvent, LearnerEventBatch
from configs.config import cfg

logger = logging.getLogger("learnflow.feature_engineering")

SEQ_LEN      = cfg.data.seq_len
NUM_FEATURES = cfg.data.n_features   # 6 base numerical features
NUM_MODULES  = cfg.data.n_modules


# ─────────────────────────────────────────────────────────────────────────────
# PER-EVENT DERIVED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _compute_derived(events: list[LearnerEvent]) -> list[dict]:
    """
    Given a chronologically sorted list of events for ONE learner,
    compute derived features row-by-row.

    Returns a list of flat dicts (one per event) containing both
    raw and derived features.
    """
    rows = []
    prev_score     = None
    score_window   = []
    engage_window  = []
    badge_count    = 0

    for i, e in enumerate(events):
        score_delta    = (e.quiz_score - prev_score) if prev_score is not None else 0.0
        score_window.append(e.quiz_score)
        engage_window.append(e.engagement_rate)

        rolling_score_3   = float(np.mean(score_window[-3:]))
        rolling_engage_3  = float(np.mean(engage_window[-3:]))
        total_attempts    = e.correct_attempts + e.incorrect_attempts
        attempt_ratio     = e.correct_attempts / max(total_attempts, 1)
        hint_rate         = e.hint_count / 10.0   # normalised to [0,1]

        if e.badge_earned:
            badge_count += 1

        rows.append({
            # Raw base features
            "quiz_score":          e.quiz_score,
            "engagement_rate":     e.engagement_rate,
            "hint_count":          float(e.hint_count),
            "session_duration":    e.session_duration,
            "correct_attempts":    float(e.correct_attempts),
            "incorrect_attempts":  float(e.incorrect_attempts),
            # Categorical
            "module_id":           e.module_id,
            # Derived
            "score_delta":         score_delta,
            "rolling_score_3":     rolling_score_3,
            "rolling_engagement_3": rolling_engage_3,
            "attempt_ratio":       attempt_ratio,
            "hint_rate":           hint_rate,
            "cumulative_badges":   float(badge_count),
            "session_number":      float(i + 1),
            # Metadata
            "learner_id":          e.learner_id,
            "timestamp":           e.timestamp,
        })
        prev_score = e.quiz_score

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

# The 6 base numerical features used by the LSTM
BASE_NUM_COLS = [
    "quiz_score", "engagement_rate", "hint_count",
    "session_duration", "correct_attempts", "incorrect_attempts",
]


def _build_sequence(
    rows: list[dict],
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a list of feature rows (for one learner) into:
      num_seq : (seq_len, 6)  raw numerical features (pre-padded with 0s)
      cat_seq : (seq_len,)    module_id integers

    Pre-padding: short sequences are padded at the FRONT with zeros so
    the most recent sessions are always at the end of the window.
    """
    num_seq = np.zeros((seq_len, len(BASE_NUM_COLS)), dtype=np.float32)
    cat_seq = np.zeros((seq_len,),                   dtype=np.int32)

    take   = min(len(rows), seq_len)
    offset = seq_len - take

    for i, row in enumerate(rows[-take:]):
        idx = offset + i
        num_seq[idx] = [row[c] for c in BASE_NUM_COLS]
        cat_seq[idx] = int(row["module_id"])

    return num_seq, cat_seq


def _compute_labels(rows: list[dict]) -> np.ndarray:
    """
    Compute target labels from the last 3 sessions of a learner's history.

      performance_score : mean quiz score of last 3 sessions
      mastery_achieved  : 1 if mean score > 65 and attempt_ratio > 0.55
      dropout_risk      : 1 if mean engagement < 0.35 and hint_rate > 0.45
    """
    last3 = rows[-3:]
    perf     = float(np.mean([r["quiz_score"]       for r in last3]))
    eng_mean = float(np.mean([r["engagement_rate"]  for r in last3]))
    hint_mean = float(np.mean([r["hint_rate"]       for r in last3]))
    ratio_mean = float(np.mean([r["attempt_ratio"]  for r in last3]))

    mastery = int(perf > 65 and ratio_mean > 0.55)
    dropout = int(eng_mean < 0.35 and hint_mean > 0.45)

    return np.array([perf, mastery, dropout], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforms a stream of LearnerEventBatches into model-ready tensors.

    Maintains a per-learner session buffer across batches so that
    streaming ingestion accumulates history correctly.

    Usage — offline
    ---------------
        fe = FeatureEngineer()
        for batch in ingester.ingest():
            clean_batch, _ = cleaner.clean(batch)
            fe.update(clean_batch)
        X_num, X_cat, y, learner_ids = fe.build_tensors()

    Usage — online (one learner at a time)
    ----------------------------------------
        num_seq, cat_seq = fe.get_sequence(learner_id)
    """

    def __init__(self, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        # learner_id → list of derived feature dicts (sorted by timestamp)
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._scaler: Optional[MinMaxScaler] = None
        self._is_fitted: bool = False

    # ── Accumulate events ─────────────────────────────────────────────────────

    def update(self, batch: LearnerEventBatch) -> None:
        """Add a batch of cleaned events to the per-learner history."""
        # Group events by learner
        by_learner: dict[str, list[LearnerEvent]] = defaultdict(list)
        for e in batch.events:
            by_learner[e.learner_id].append(e)

        for lid, events in by_learner.items():
            # Sort new events by timestamp
            events = sorted(events, key=lambda e: e.timestamp)
            new_rows = _compute_derived(events)
            self._history[lid].extend(new_rows)
            # Keep only the last seq_len * 2 rows to bound memory
            self._history[lid] = self._history[lid][-(self.seq_len * 2):]

    # ── Per-learner sequence ──────────────────────────────────────────────────

    def get_sequence(self, learner_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the raw (unscaled) feature sequence for a single learner.
        Returns zero-padded arrays if fewer than seq_len sessions are known.
        """
        rows = self._history.get(learner_id, [])
        return _build_sequence(rows, self.seq_len)

    # ── Full tensor build (offline) ───────────────────────────────────────────

    def build_tensors(
        self,
        fit_scaler: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], MinMaxScaler]:
        """
        Build model-ready tensors from all accumulated histories.

        Returns
        -------
        X_num       : (N, seq_len, 6)  scaled numerical features
        X_cat       : (N, seq_len)     module_id integers
        y           : (N, 3)           [performance, mastery, dropout]
        learner_ids : list of N learner ID strings
        scaler      : fitted MinMaxScaler (save with the checkpoint)
        """
        if not self._history:
            raise ValueError("No events have been ingested yet. Call update() first.")

        learner_ids = sorted(self._history.keys())
        num_list, cat_list, y_list = [], [], []

        for lid in learner_ids:
            rows = self._history[lid]
            if not rows:
                continue
            num_seq, cat_seq = _build_sequence(rows, self.seq_len)
            labels           = _compute_labels(rows)
            num_list.append(num_seq)
            cat_list.append(cat_seq)
            y_list.append(labels)

        X_num = np.stack(num_list).astype(np.float32)   # (N, T, 6)
        X_cat = np.stack(cat_list).astype(np.int32)     # (N, T)
        y     = np.stack(y_list).astype(np.float32)     # (N, 3)

        # ── Fit / apply scaler ───────────────────────────────────────────
        N, T, F = X_num.shape
        flat    = X_num.reshape(-1, F)

        if fit_scaler or self._scaler is None:
            self._scaler    = MinMaxScaler()
            flat_scaled     = self._scaler.fit_transform(flat)
            self._is_fitted = True
        else:
            flat_scaled = self._scaler.transform(flat)

        X_num_scaled = flat_scaled.reshape(N, T, F).astype(np.float32)

        logger.info(
            "FeatureEngineer: built tensors for %d learners  "
            "mastery_rate=%.1f%%  dropout_rate=%.1f%%",
            N,
            100 * y[:, 1].mean(),
            100 * y[:, 2].mean(),
        )

        return X_num_scaled, X_cat, y, learner_ids, self._scaler

    # ── Scaler persistence ────────────────────────────────────────────────────

    def save_scaler(self, path) -> None:
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted yet. Call build_tensors() first.")
        import numpy as np
        np.save(str(path), {
            "scale_":      self._scaler.scale_,
            "min_":        self._scaler.min_,
            "data_min_":   self._scaler.data_min_,
            "data_max_":   self._scaler.data_max_,
            "data_range_": self._scaler.data_range_,
        })
        logger.info("Scaler saved to %s", path)

    def load_scaler(self, path) -> None:
        params = np.load(str(path), allow_pickle=True).item()
        self._scaler = MinMaxScaler()
        for k, v in params.items():
            setattr(self._scaler, k, v)
        self._scaler.n_features_in_ = NUM_FEATURES
        self._scaler.n_samples_seen_ = 1
        self._is_fitted = True
        logger.info("Scaler loaded from %s", path)

    @property
    def n_learners(self) -> int:
        return len(self._history)

    @property
    def learner_ids(self) -> list[str]:
        return list(self._history.keys())