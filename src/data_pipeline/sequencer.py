"""
src/data_pipeline/sequencer.py
================================
Converts a processed learner DataFrame into padded, windowed sequence
tensors ready for LSTM training.

Responsibilities
----------------
• Group events by learner and sort chronologically
• Apply sliding-window extraction (for learners with > seq_len sessions)
• Stratified train / val / test split preserving class balance
• Persist processed tensors as .npy files for fast reloading
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import cfg

logger = logging.getLogger("learnflow.sequencer")

SEQ_LEN     = cfg.data.seq_len
NUM_FEATURES = cfg.data.n_features

BASE_NUM_COLS = [
    "quiz_score", "engagement_rate", "hint_count",
    "session_duration", "correct_attempts", "incorrect_attempts",
]


class Sequencer:
    """
    Builds padded (seq_len, n_features) sequences from a flat event DataFrame.

    Sliding-window extraction
    -------------------------
    For a learner with M sessions (M > seq_len):
      produces M - seq_len + 1 overlapping windows
      the label for each window is derived from the final 3 sessions
      of that window.

    This greatly increases the effective training set size.

    Parameters
    ----------
    seq_len       : fixed sequence length (default from config)
    use_sliding   : enable sliding-window extraction
    stride        : step between consecutive windows
    """

    def __init__(
        self,
        seq_len:     int  = SEQ_LEN,
        use_sliding: bool = True,
        stride:      int  = 1,
    ):
        self.seq_len     = seq_len
        self.use_sliding = use_sliding
        self.stride      = stride

    # ── Public API ────────────────────────────────────────────────────────────

    def build_from_df(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Build tensors from a flat DataFrame.

        Required columns
        ----------------
        learner_id, module_id, quiz_score, engagement_rate,
        hint_count, session_duration, correct_attempts, incorrect_attempts

        Returns
        -------
        X_num       : (N, seq_len, 6)
        X_cat       : (N, seq_len)
        y           : (N, 3)  [performance, mastery, dropout]
        learner_ids : (N,)    learner ID for each sample
        """
        required = set(BASE_NUM_COLS + ["learner_id", "module_id"])
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        num_seqs, cat_seqs, labels, ids = [], [], [], []

        for lid, group in df.groupby("learner_id"):
            group = group.sort_values("timestamp") if "timestamp" in group.columns \
                    else group.reset_index(drop=True)

            windows = self._extract_windows(group)
            for num_seq, cat_seq, y in windows:
                num_seqs.append(num_seq)
                cat_seqs.append(cat_seq)
                labels.append(y)
                ids.append(str(lid))

        if not num_seqs:
            raise ValueError("No sequences could be built from the DataFrame.")

        X_num = np.stack(num_seqs).astype(np.float32)
        X_cat = np.stack(cat_seqs).astype(np.int32)
        y     = np.stack(labels).astype(np.float32)

        logger.info(
            "Sequencer: %d sequences from %d learners  "
            "mastery=%.1f%%  dropout=%.1f%%",
            len(X_num), df["learner_id"].nunique(),
            100 * y[:, 1].mean(),
            100 * y[:, 2].mean(),
        )
        return X_num, X_cat, y, ids

    def split(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        y:     np.ndarray,
        train_ratio: float = None,
        val_ratio:   float = None,
        seed:        int   = None,
    ) -> dict[str, tuple]:
        """
        Stratified train / val / test split on the dropout label.

        Returns
        -------
        dict with keys 'train', 'val', 'test',
        each mapping to (X_num, X_cat, y) tuples.
        """
        train_r = train_ratio or cfg.data.train_ratio
        val_r   = val_ratio   or cfg.data.val_ratio
        seed_   = seed        or cfg.data.synthetic_seed

        stratify = y[:, 2].astype(int)   # dropout label

        # First split off test set
        test_size = 1.0 - train_r - val_r
        X_num_tv, X_num_test, X_cat_tv, X_cat_test, y_tv, y_test = train_test_split(
            X_num, X_cat, y,
            test_size=test_size, random_state=seed_,
            stratify=stratify,
        )

        # Then split validation from training
        val_fraction = val_r / (train_r + val_r)
        X_num_tr, X_num_val, X_cat_tr, X_cat_val, y_tr, y_val = train_test_split(
            X_num_tv, X_cat_tv, y_tv,
            test_size=val_fraction, random_state=seed_,
            stratify=y_tv[:, 2].astype(int),
        )

        logger.info(
            "Split: train=%d  val=%d  test=%d",
            len(X_num_tr), len(X_num_val), len(X_num_test),
        )

        return {
            "train": (X_num_tr,   X_cat_tr,   y_tr),
            "val":   (X_num_val,  X_cat_val,  y_val),
            "test":  (X_num_test, X_cat_test, y_test),
        }

    def save(self, path: Path, X_num, X_cat, y, split: str = "train") -> None:
        """Save tensors as .npy files under path/split/."""
        out = Path(path) / split
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X_num.npy", X_num)
        np.save(out / "X_cat.npy", X_cat)
        np.save(out / "y.npy",     y)
        logger.info("Saved %s tensors to %s", split, out)

    def load(self, path: Path, split: str = "train") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load previously saved tensors."""
        out = Path(path) / split
        X_num = np.load(out / "X_num.npy")
        X_cat = np.load(out / "X_cat.npy")
        y     = np.load(out / "y.npy")
        logger.info("Loaded %s tensors from %s  shape=%s", split, out, X_num.shape)
        return X_num, X_cat, y

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_windows(
        self,
        group: pd.DataFrame,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract (possibly overlapping) sequence windows from one learner's
        sorted event rows.
        """
        rows = group[BASE_NUM_COLS + ["module_id"]].values
        M    = len(rows)

        windows = []

        if M < self.seq_len or not self.use_sliding:
            # Single window: pre-pad with zeros
            num_seq, cat_seq = self._pad(rows, M)
            y_labels         = self._label(rows)
            windows.append((num_seq, cat_seq, y_labels))
        else:
            # Sliding windows
            starts = range(0, M - self.seq_len + 1, self.stride)
            for start in starts:
                chunk    = rows[start : start + self.seq_len]
                num_seq  = chunk[:, :NUM_FEATURES].astype(np.float32)
                cat_seq  = chunk[:, -1].astype(np.int32)
                y_labels = self._label(chunk)
                windows.append((num_seq, cat_seq, y_labels))

        return windows

    def _pad(
        self,
        rows: np.ndarray,
        M: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_seq = np.zeros((self.seq_len, NUM_FEATURES), dtype=np.float32)
        cat_seq = np.zeros((self.seq_len,),              dtype=np.int32)
        offset  = self.seq_len - min(M, self.seq_len)

        take = min(M, self.seq_len)
        num_seq[offset:] = rows[-take:, :NUM_FEATURES]
        cat_seq[offset:] = rows[-take:, -1].astype(int)
        return num_seq, cat_seq

    @staticmethod
    def _label(rows: np.ndarray) -> np.ndarray:
        """Derive target labels from the last 3 rows of a window."""
        last3      = rows[-3:]
        perf       = float(np.mean(last3[:, 0]))                      # quiz_score
        eng_mean   = float(np.mean(last3[:, 1]))                      # engagement_rate
        hint_mean  = float(np.mean(last3[:, 2])) / 10.0               # hint_rate
        total_att  = last3[:, 4] + last3[:, 5]                        # correct + incorrect
        ratio_mean = float(np.mean(
            last3[:, 4] / np.maximum(total_att, 1)                    # attempt_ratio
        ))

        mastery = int(perf > 65 and ratio_mean > 0.55)
        dropout = int(eng_mean < 0.35 and hint_mean > 0.45)

        return np.array([perf, mastery, dropout], dtype=np.float32)