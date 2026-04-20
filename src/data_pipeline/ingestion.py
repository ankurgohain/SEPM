"""
src/data_pipeline/ingestion.py
==============================
Multi-source ingestion layer.  Reads raw learner event data from:

  ┌─────────────────────────────────────────────────────────┐
  │  Source            Class                   Mode          │
  │  ──────────────────────────────────────────────────────  │
  │  CSV / TSV file    CSVIngester             batch         │
  │  Parquet file      ParquetIngester         batch         │
  │  JSON-lines file   JSONLinesIngester       batch         │
  │  Kafka topic       KafkaIngester           streaming     │
  │  Dict / list       InMemoryIngester        programmatic  │
  └─────────────────────────────────────────────────────────┘

All ingesters yield LearnerEventBatch objects so that downstream
cleaning and feature engineering are source-agnostic.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd

from .schema import LearnerEvent, LearnerEventBatch, MODULES, MODULE_TO_ID
from src.configs.config import cfg

logger = logging.getLogger("learnflow.ingestion")


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BaseIngester(ABC):
    """Abstract base for all data sources."""

    @abstractmethod
    def ingest(self) -> Generator[LearnerEventBatch, None, None]:
        """Yield successive batches of validated LearnerEvents."""

    @staticmethod
    def _parse_row(row: dict, source: str = "unknown") -> Optional[LearnerEvent]:
        """
        Try to coerce a raw dict into a LearnerEvent.
        Returns None on validation failure (caller decides to skip or abort).
        """
        try:
            # Normalise common alternate column names
            row = _normalise_columns(row)
            return LearnerEvent(**row)
        except Exception as exc:
            logger.debug("Skipping malformed row: %s  →  %s", row, exc)
            return None

    @staticmethod
    def _chunk(lst: list, n: int) -> Generator[list, None, None]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NAME NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

_COLUMN_ALIASES: dict[str, str] = {
    # quiz score variants
    "score":             "quiz_score",
    "quiz":              "quiz_score",
    "grade":             "quiz_score",
    "marks":             "quiz_score",
    # engagement
    "engagement":        "engagement_rate",
    "on_task_rate":      "engagement_rate",
    "time_on_task":      "engagement_rate",
    # hints
    "hints":             "hint_count",
    "hint_used":         "hint_count",
    "hints_used":        "hint_count",
    # duration
    "duration":          "session_duration",
    "time_spent":        "session_duration",
    "session_time":      "session_duration",
    "minutes":           "session_duration",
    # attempts
    "correct":           "correct_attempts",
    "num_correct":       "correct_attempts",
    "incorrect":         "incorrect_attempts",
    "num_incorrect":     "incorrect_attempts",
    "wrong":             "incorrect_attempts",
    # module
    "course":            "module",
    "course_id":         "module",
    "module_name":       "module",
    "subject":           "module",
    # learner
    "user_id":           "learner_id",
    "student_id":        "learner_id",
    "learner":           "learner_id",
    # timestamp
    "date":              "timestamp",
    "event_time":        "timestamp",
    "session_date":      "timestamp",
    "created_at":        "timestamp",
}


def _normalise_columns(row: dict) -> dict:
    """Apply alias mapping and coerce dtypes."""
    out = {}
    for k, v in row.items():
        canonical = _COLUMN_ALIASES.get(k.lower().strip(), k.lower().strip())
        out[canonical] = v

    # Coerce timestamp
    if "timestamp" in out and not isinstance(out["timestamp"], datetime):
        ts = out["timestamp"]
        if isinstance(ts, (int, float)):
            out["timestamp"] = datetime.utcfromtimestamp(float(ts))
        else:
            try:
                out["timestamp"] = datetime.fromisoformat(str(ts))
            except Exception:
                out["timestamp"] = datetime.utcnow()

    # Coerce numeric fields
    for num_col in ("quiz_score", "engagement_rate", "session_duration"):
        if num_col in out:
            try:
                out[num_col] = float(out[num_col])
            except (ValueError, TypeError):
                out[num_col] = 0.0

    for int_col in ("hint_count", "correct_attempts", "incorrect_attempts", "module_id"):
        if int_col in out:
            try:
                out[int_col] = int(float(out[int_col]))
            except (ValueError, TypeError):
                out[int_col] = 0

    # Resolve module slug from module_id if module not present
    if "module" not in out and "module_id" in out:
        mid = int(out["module_id"])
        out["module"] = MODULES[mid] if 0 <= mid < len(MODULES) else "python_basics"

    # Default badge_earned
    if "badge_earned" not in out:
        out["badge_earned"] = False
    elif isinstance(out["badge_earned"], str):
        out["badge_earned"] = out["badge_earned"].lower() in ("1", "true", "yes")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CSV INGESTER
# ─────────────────────────────────────────────────────────────────────────────

class CSVIngester(BaseIngester):
    """
    Ingest a CSV / TSV file containing one row per learning session.

    The file must have a header row.  Column names are auto-mapped via
    _normalise_columns so files from common LMS platforms (Moodle,
    Canvas, Blackboard) are handled without pre-processing.

    Parameters
    ----------
    path        : path to the CSV file
    delimiter   : column delimiter (default ',')
    chunk_size  : number of rows per yielded batch
    encoding    : file encoding
    """

    def __init__(
        self,
        path:        Path | str,
        delimiter:   str  = None,
        chunk_size:  int  = None,
        encoding:    str  = None,
    ):
        self.path       = Path(path)
        self.delimiter  = delimiter  or cfg.data.csv_delimiter
        self.chunk_size = chunk_size or cfg.data.ingest_chunk_size
        self.encoding   = encoding   or cfg.data.csv_encoding

    def ingest(self) -> Generator[LearnerEventBatch, None, None]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        logger.info("CSVIngester: reading %s", self.path)
        total_read = 0
        total_ok   = 0

        with open(self.path, encoding=self.encoding, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter)
            buffer: list[LearnerEvent] = []

            for raw_row in reader:
                total_read += 1
                event = self._parse_row(dict(raw_row), source=str(self.path))
                if event:
                    buffer.append(event)
                    total_ok += 1

                if len(buffer) >= self.chunk_size:
                    yield LearnerEventBatch(events=buffer, source=str(self.path))
                    buffer = []

            if buffer:
                yield LearnerEventBatch(events=buffer, source=str(self.path))

        logger.info(
            "CSVIngester: %d rows read, %d valid (%.1f%%)",
            total_read, total_ok,
            100 * total_ok / max(total_read, 1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PARQUET INGESTER
# ─────────────────────────────────────────────────────────────────────────────

class ParquetIngester(BaseIngester):
    """
    Ingest a Parquet file (or directory of partitioned Parquet files).

    Requires pandas + pyarrow.
    """

    def __init__(self, path: Path | str, chunk_size: int = None):
        self.path       = Path(path)
        self.chunk_size = chunk_size or cfg.data.ingest_chunk_size

    def ingest(self) -> Generator[LearnerEventBatch, None, None]:
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet path not found: {self.path}")

        logger.info("ParquetIngester: reading %s", self.path)
        df = pd.read_parquet(self.path)

        total_ok = 0
        for chunk_df in self._df_chunks(df, self.chunk_size):
            events = []
            for _, row in chunk_df.iterrows():
                event = self._parse_row(row.to_dict(), source=str(self.path))
                if event:
                    events.append(event)
                    total_ok += 1
            if events:
                yield LearnerEventBatch(events=events, source=str(self.path))

        logger.info("ParquetIngester: %d valid events loaded", total_ok)

    @staticmethod
    def _df_chunks(df: pd.DataFrame, n: int) -> Generator[pd.DataFrame, None, None]:
        for i in range(0, len(df), n):
            yield df.iloc[i : i + n]


# ─────────────────────────────────────────────────────────────────────────────
# JSON-LINES INGESTER
# ─────────────────────────────────────────────────────────────────────────────

class JSONLinesIngester(BaseIngester):
    """
    Ingest a newline-delimited JSON file (.jsonl).
    Each line must be a JSON object representing one session.
    """

    def __init__(self, path: Path | str, chunk_size: int = None):
        self.path       = Path(path)
        self.chunk_size = chunk_size or cfg.data.ingest_chunk_size

    def ingest(self) -> Generator[LearnerEventBatch, None, None]:
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.path}")

        logger.info("JSONLinesIngester: reading %s", self.path)
        buffer: list[LearnerEvent] = []
        total_read = total_ok = 0

        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total_read += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON line: %s", line[:80])
                    continue

                event = self._parse_row(raw, source=str(self.path))
                if event:
                    buffer.append(event)
                    total_ok += 1

                if len(buffer) >= self.chunk_size:
                    yield LearnerEventBatch(events=buffer, source=str(self.path))
                    buffer = []

            if buffer:
                yield LearnerEventBatch(events=buffer, source=str(self.path))

        logger.info(
            "JSONLinesIngester: %d lines, %d valid (%.1f%%)",
            total_read, total_ok, 100 * total_ok / max(total_read, 1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# KAFKA INGESTER  (streaming, real-time)
# ─────────────────────────────────────────────────────────────────────────────

class KafkaIngester(BaseIngester):
    """
    Consume learner events from a Kafka topic in real-time.

    Requires `kafka-python` to be installed.  If kafka-python is not
    present the ingester raises ImportError with a helpful message.

    The consumer runs until `max_events` are consumed or `timeout_ms`
    elapses with no new messages.

    Parameters
    ----------
    bootstrap_servers : Kafka broker list (comma-separated string)
    topic             : Kafka topic name
    group_id          : consumer group
    batch_size        : yield a batch every N events
    timeout_ms        : poll timeout (ms); 0 = non-blocking
    max_events        : stop after this many events (None = run forever)
    """

    def __init__(
        self,
        bootstrap_servers: str  = None,
        topic:             str  = None,
        group_id:          str  = None,
        batch_size:        int  = None,
        timeout_ms:        int  = 1000,
        max_events:        Optional[int] = None,
    ):
        self.bootstrap_servers = bootstrap_servers or cfg.data.kafka_bootstrap
        self.topic             = topic             or cfg.data.kafka_topic
        self.group_id          = group_id          or cfg.data.kafka_group_id
        self.batch_size        = batch_size        or cfg.data.kafka_batch_size
        self.timeout_ms        = timeout_ms
        self.max_events        = max_events

    def ingest(self) -> Generator[LearnerEventBatch, None, None]:
        try:
            from kafka import KafkaConsumer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "kafka-python is required for Kafka ingestion. "
                "Install it with: pip install kafka-python"
            ) from e

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers.split(","),
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )

        logger.info(
            "KafkaIngester: consuming topic=%s bootstrap=%s",
            self.topic, self.bootstrap_servers,
        )

        buffer:        list[LearnerEvent] = []
        total_consumed = 0

        try:
            for msg in consumer:
                raw   = msg.value
                event = self._parse_row(raw, source=f"kafka:{self.topic}")
                if event:
                    buffer.append(event)
                    total_consumed += 1

                if len(buffer) >= self.batch_size:
                    yield LearnerEventBatch(events=buffer, source=f"kafka:{self.topic}")
                    buffer = []

                if self.max_events and total_consumed >= self.max_events:
                    break
        finally:
            if buffer:
                yield LearnerEventBatch(events=buffer, source=f"kafka:{self.topic}")
            consumer.close()
            logger.info("KafkaIngester: consumed %d events", total_consumed)


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY INGESTER  (for testing and API-push scenarios)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryIngester(BaseIngester):
    """
    Wrap a list of dicts or a DataFrame as an ingestion source.
    Useful for unit tests and for ingesting data pushed via REST.
    """

    def __init__(self, records: list[dict] | pd.DataFrame, chunk_size: int = None):
        if isinstance(records, pd.DataFrame):
            self._records = records.to_dict(orient="records")
        else:
            self._records = list(records)
        self.chunk_size = chunk_size or cfg.data.ingest_chunk_size

    def ingest(self) -> Generator[LearnerEventBatch, None, None]:
        events = []
        for row in self._records:
            event = self._parse_row(dict(row), source="in_memory")
            if event:
                events.append(event)

        for chunk in self._chunk(events, self.chunk_size):
            yield LearnerEventBatch(events=chunk, source="in_memory")


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA INGESTER  (generates data on-the-fly for demos / CI)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticIngester(BaseIngester):
    """
    Generate synthetic learner events that pass through the full pipeline.
    Wraps LearnerDataGenerator from lstm_model.py as an ingestion source.

    Useful for:
      • Integration tests
      • Pipeline smoke-tests in CI
      • Dashboard demos without a real LMS
    """

    def __init__(self, n_learners: int = None, seq_len: int = None, seed: int = None):
        self.n_learners = n_learners or cfg.data.n_synthetic_learners
        self.seq_len    = seq_len    or cfg.data.seq_len
        self.seed       = seed       or cfg.data.synthetic_seed

    def ingest(self) -> Generator[LearnerEventBatch, None, None]:
        logger.info(
            "SyntheticIngester: generating %d learners × %d sessions",
            self.n_learners, self.seq_len,
        )
        np.random.seed(self.seed)
        events: list[LearnerEvent] = []
        base_time = datetime(2024, 1, 1)

        for lid in range(self.n_learners):
            learner_id  = f"learner_{lid:05d}"
            ability     = float(np.clip(np.random.beta(2, 2), 0.05, 0.95))
            module_idx  = np.random.randint(0, len(MODULES))

            for t in range(self.seq_len):
                quiz    = float(np.clip(ability * 80 + np.random.normal(0, 12) + t * ability * 1.5, 10, 100))
                eng     = float(np.clip(ability * 0.8 + np.random.normal(0, 0.12), 0.05, 1.0))
                hints   = int(np.clip(np.random.poisson((1 - ability) * 5), 0, 10))
                dur     = float(np.clip(np.random.normal(40 + ability * 30, 10), 5, 120))
                correct = int(np.clip(np.random.poisson(ability * 15), 0, 20))
                wrong   = int(np.clip(np.random.poisson((1 - ability) * 8), 0, 20))

                if t > 0 and t % 4 == 0:
                    module_idx = min(module_idx + 1, len(MODULES) - 1)

                ts = base_time.replace(
                    hour=np.random.randint(8, 22),
                    minute=np.random.randint(0, 59),
                )

                events.append(LearnerEvent(
                    learner_id         = learner_id,
                    timestamp          = ts,
                    module             = MODULES[module_idx],
                    quiz_score         = round(quiz, 2),
                    engagement_rate    = round(eng, 3),
                    hint_count         = hints,
                    session_duration   = round(dur, 1),
                    correct_attempts   = correct,
                    incorrect_attempts = wrong,
                    badge_earned       = bool(ability > 0.7 and t == self.seq_len - 1),
                ))

        logger.info("SyntheticIngester: %d events generated", len(events))
        for chunk in self._chunk(events, cfg.data.ingest_chunk_size):
            yield LearnerEventBatch(events=chunk, source="synthetic")