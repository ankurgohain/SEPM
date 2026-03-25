"""
src/data_pipeline/cleaner.py
=============================
Cleans raw LearnerEvent batches before feature engineering.

Operations (in order)
─────────────────────
1. Drop events with null learner_id or timestamp
2. Clip numerical outliers to valid domain ranges
3. Fill missing hint_count / attempt counts with 0
4. Detect and remove exact duplicate events (same learner + timestamp)
5. Sort each learner's events by timestamp (chronological order)
6. Emit a CleanReport with per-batch quality statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.data_pipeline.schema import LearnerEvent, LearnerEventBatch

logger = logging.getLogger("learnflow.cleaner")

# Domain bounds — any value outside these is clipped
BOUNDS: dict[str, tuple[float, float]] = {
    "quiz_score":          (0.0,   100.0),
    "engagement_rate":     (0.0,     1.0),
    "hint_count":          (0.0,    10.0),
    "session_duration":    (1.0,   240.0),
    "correct_attempts":    (0.0,    50.0),
    "incorrect_attempts":  (0.0,    50.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanReport:
    """Summary statistics produced by the cleaner for one batch."""

    total_in:        int = 0
    dropped_null:    int = 0
    dropped_dup:     int = 0
    clipped:         int = 0    # events where at least one value was clipped
    total_out:       int = 0

    @property
    def drop_rate(self) -> float:
        return (self.dropped_null + self.dropped_dup) / max(self.total_in, 1)

    def __str__(self) -> str:
        return (
            f"CleanReport: in={self.total_in}  out={self.total_out}  "
            f"null_drops={self.dropped_null}  dup_drops={self.dropped_dup}  "
            f"clipped={self.clipped}  drop_rate={self.drop_rate:.1%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLEANER
# ─────────────────────────────────────────────────────────────────────────────

class EventCleaner:
    """
    Stateless cleaner — all logic is pure functions of the input batch.

    Usage
    -----
        cleaner = EventCleaner()
        clean_batch, report = cleaner.clean(raw_batch)
    """

    def clean(
        self,
        batch: LearnerEventBatch,
    ) -> tuple[LearnerEventBatch, CleanReport]:
        """
        Clean a batch of LearnerEvents.

        Returns
        -------
        clean_batch : LearnerEventBatch with invalid events removed
        report      : CleanReport with quality statistics
        """
        report = CleanReport(total_in=batch.size)
        events = list(batch.events)

        # ── 1. Drop null / empty learner_id or missing timestamp ─────────
        before = len(events)
        events = [
            e for e in events
            if e.learner_id and e.learner_id.strip() and e.timestamp
        ]
        report.dropped_null += before - len(events)

        # ── 2. Clip numerical outliers ────────────────────────────────────
        clipped_events = []
        for e in events:
            clipped, was_clipped = _clip_event(e)
            clipped_events.append(clipped)
            if was_clipped:
                report.clipped += 1
        events = clipped_events

        # ── 3. Remove exact duplicates (learner_id + timestamp) ──────────
        before = len(events)
        seen: set[tuple] = set()
        deduped = []
        for e in events:
            key = (e.learner_id, e.timestamp.isoformat())
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        report.dropped_dup += before - len(deduped)
        events = deduped

        # ── 4. Sort each learner's events chronologically ─────────────────
        events = sorted(events, key=lambda e: (e.learner_id, e.timestamp))

        report.total_out = len(events)
        logger.debug(str(report))

        clean_batch = LearnerEventBatch(
            events=events,
            source=batch.source,
            ingested_at=batch.ingested_at,
        )
        return clean_batch, report


def _clip_event(event: LearnerEvent) -> tuple[LearnerEvent, bool]:
    """
    Clip a single event's numerical fields to domain bounds.
    Returns the (possibly modified) event and a boolean indicating
    whether any clipping occurred.
    """
    was_clipped = False
    kwargs = event.model_dump()

    for col, (lo, hi) in BOUNDS.items():
        val = kwargs.get(col)
        if val is None:
            # Fill missing with safe defaults
            kwargs[col] = lo
            was_clipped = True
        elif float(val) < lo or float(val) > hi:
            kwargs[col] = float(np.clip(val, lo, hi))
            was_clipped = True

    if was_clipped:
        return LearnerEvent(**kwargs), True
    return event, False


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME-LEVEL CLEANER  (for batch/offline pipelines)
# ─────────────────────────────────────────────────────────────────────────────

class DataFrameCleaner:
    """
    Cleans a raw pandas DataFrame of session records.
    Faster than EventCleaner for large offline batches.

    Parameters
    ----------
    clip_outliers   : apply domain-bound clipping
    drop_duplicates : drop exact (learner_id, timestamp) duplicates
    fill_na         : fill numeric NaN with 0
    """

    NUMERIC_COLS = list(BOUNDS.keys())
    REQUIRED_COLS = {"learner_id", "timestamp", "module",
                     "quiz_score", "engagement_rate", "session_duration"}

    def __init__(
        self,
        clip_outliers:   bool = True,
        drop_duplicates: bool = True,
        fill_na:         bool = True,
    ):
        self.clip_outliers   = clip_outliers
        self.drop_duplicates = drop_duplicates
        self.fill_na         = fill_na

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
        report = CleanReport(total_in=len(df))
        df = df.copy()

        # ── Check required columns ────────────────────────────────────────
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # ── Drop nulls in key columns ─────────────────────────────────────
        before = len(df)
        df = df.dropna(subset=["learner_id", "timestamp"])
        df = df[df["learner_id"].astype(str).str.strip() != ""]
        report.dropped_null = before - len(df)

        # ── Fill numeric NaN ──────────────────────────────────────────────
        if self.fill_na:
            for col in self.NUMERIC_COLS:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)

        # ── Clip outliers ─────────────────────────────────────────────────
        if self.clip_outliers:
            for col, (lo, hi) in BOUNDS.items():
                if col in df.columns:
                    original = df[col].copy()
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    report.clipped += int((df[col] != original).sum())

        # ── Drop duplicates ────────────────────────────────────────────────
        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates(subset=["learner_id", "timestamp"])
            report.dropped_dup = before - len(df)

        # ── Sort chronologically per learner ───────────────────────────────
        df = df.sort_values(["learner_id", "timestamp"]).reset_index(drop=True)

        report.total_out = len(df)
        logger.info(str(report))
        return df, report