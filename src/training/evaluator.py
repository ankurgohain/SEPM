"""
src/training/evaluator.py
==========================
Standalone evaluator for trained LSTM models.

Provides per-head metrics (RMSE, AUC, F1, confusion matrix) and
exports results as a structured dict suitable for MLflow / JSON logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)

logger = logging.getLogger("learnflow.evaluator")


@dataclass
class EvalResult:
    """Container for per-head evaluation metrics."""

    # Performance score (regression)
    perf_rmse: float = 0.0
    perf_mae:  float = 0.0

    # Mastery classification
    mastery_auc:  float = 0.0
    mastery_ap:   float = 0.0   # average precision (PR-AUC)
    mastery_f1:   float = 0.0
    mastery_prec: float = 0.0
    mastery_rec:  float = 0.0

    # Dropout classification
    dropout_auc:  float = 0.0
    dropout_ap:   float = 0.0
    dropout_f1:   float = 0.0
    dropout_prec: float = 0.0
    dropout_rec:  float = 0.0

    # Confusion matrices (serialised as lists)
    mastery_cm: list = field(default_factory=list)
    dropout_cm: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"EvalResult | "
            f"perf_rmse={self.perf_rmse:.3f} | "
            f"mastery_auc={self.mastery_auc:.4f} | "
            f"dropout_auc={self.dropout_auc:.4f}"
        )


class ModelEvaluator:
    """
    Evaluates a trained LSTM model on any (X_num, X_cat, y) split.

    Parameters
    ----------
    threshold_mastery : decision threshold for mastery binary label
    threshold_dropout : decision threshold for dropout binary label
    """

    def __init__(
        self,
        threshold_mastery: float = 0.50,
        threshold_dropout: float = 0.50,
    ):
        self.threshold_mastery = threshold_mastery
        self.threshold_dropout = threshold_dropout

    def evaluate(
        self,
        model,
        X_num:  np.ndarray,
        X_cat:  np.ndarray,
        y_true: np.ndarray,
        split_name: str = "Evaluation",
        verbose: bool = True,
    ) -> EvalResult:
        """
        Run inference and compute all metrics.

        Parameters
        ----------
        model      : trained Keras model with multi-output dict
        X_num      : (N, T, 6) numerical features
        X_cat      : (N, T)    module ids
        y_true     : (N, 3)    [performance, mastery, dropout]
        split_name : label for the printed report
        verbose    : print the full report

        Returns
        -------
        EvalResult dataclass
        """
        preds = model.predict([X_num, X_cat], verbose=0)

        perf_pred    = preds["performance_score"].squeeze()
        mastery_pred = preds["mastery_prob"].squeeze()
        dropout_pred = preds["dropout_risk"].squeeze()

        perf_true    = y_true[:, 0]
        mastery_true = y_true[:, 1].astype(int)
        dropout_true = y_true[:, 2].astype(int)

        mastery_bin = (mastery_pred >= self.threshold_mastery).astype(int)
        dropout_bin = (dropout_pred >= self.threshold_dropout).astype(int)

        # ── Regression metrics ──────────────────────────────────────────
        rmse = float(np.sqrt(mean_squared_error(perf_true, perf_pred)))
        mae  = float(mean_absolute_error(perf_true, perf_pred))

        # ── Classification metrics ──────────────────────────────────────
        def _safe_metric(fn, y_t, y_p, **kw):
            try:
                return float(fn(y_t, y_p, **kw))
            except (ValueError, TypeError):
                return float("nan")

        mastery_auc  = _safe_metric(roc_auc_score,           mastery_true, mastery_pred)
        mastery_ap   = _safe_metric(average_precision_score, mastery_true, mastery_pred)
        dropout_auc  = _safe_metric(roc_auc_score,           dropout_true, dropout_pred)
        dropout_ap   = _safe_metric(average_precision_score, dropout_true, dropout_pred)

        def _f1_prec_rec(y_t, y_p):
            try:
                rep = classification_report(y_t, y_p, output_dict=True, zero_division=0)
                pos = rep.get("1", rep.get("1.0", {}))
                return (
                    float(pos.get("f1-score", 0)),
                    float(pos.get("precision", 0)),
                    float(pos.get("recall", 0)),
                )
            except Exception:
                return 0.0, 0.0, 0.0

        m_f1, m_prec, m_rec = _f1_prec_rec(mastery_true, mastery_bin)
        d_f1, d_prec, d_rec = _f1_prec_rec(dropout_true, dropout_bin)

        m_cm = confusion_matrix(mastery_true, mastery_bin).tolist()
        d_cm = confusion_matrix(dropout_true, dropout_bin).tolist()

        result = EvalResult(
            perf_rmse    = rmse,
            perf_mae     = mae,
            mastery_auc  = mastery_auc,
            mastery_ap   = mastery_ap,
            mastery_f1   = m_f1,
            mastery_prec = m_prec,
            mastery_rec  = m_rec,
            dropout_auc  = dropout_auc,
            dropout_ap   = dropout_ap,
            dropout_f1   = d_f1,
            dropout_prec = d_prec,
            dropout_rec  = d_rec,
            mastery_cm   = m_cm,
            dropout_cm   = d_cm,
        )

        if verbose:
            self._print_report(result, mastery_true, mastery_bin,
                               dropout_true, dropout_bin, split_name)

        logger.info("%s  %s", split_name, result)
        return result

    # ── Pretty-print ──────────────────────────────────────────────────────────

    @staticmethod
    def _print_report(
        r: EvalResult,
        mastery_true, mastery_bin,
        dropout_true, dropout_bin,
        split_name: str,
    ) -> None:
        sep = "═" * 56
        print(f"\n{sep}")
        print(f"  {split_name} — LSTM Model Evaluation")
        print(sep)
        print(f"\n  ► Performance Score  (regression)")
        print(f"    RMSE : {r.perf_rmse:>7.3f}  (scale 0–100)")
        print(f"    MAE  : {r.perf_mae:>7.3f}")
        print(f"\n  ► Mastery Probability  (classification)")
        print(f"    ROC-AUC : {r.mastery_auc:>6.4f}")
        print(f"    PR-AUC  : {r.mastery_ap:>6.4f}")
        print(
            classification_report(
                mastery_true, mastery_bin,
                target_names=["Not Mastered", "Mastered"],
                zero_division=0,
            )
        )
        print(f"  ► Dropout Risk  (classification)")
        print(f"    ROC-AUC : {r.dropout_auc:>6.4f}")
        print(f"    PR-AUC  : {r.dropout_ap:>6.4f}")
        print(
            classification_report(
                dropout_true, dropout_bin,
                target_names=["Retained", "At Risk"],
                zero_division=0,
            )
        )
        print(sep + "\n")