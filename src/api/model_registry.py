"""
model_registry.py
=================
Singleton that loads the trained LSTM model + scaler at startup
and exposes a thread-safe predict() interface to all route handlers.

The registry is injected via FastAPI dependency injection so that
tests can swap in a mock without touching global state.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("learnflow.registry")

# ── Try importing the real model; fall back to a stub for environments
#    where TensorFlow is not installed (e.g. tests, CI). ─────────────
try:
    import tensorflow as tf
    from keras._tf_keras import keras
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    logger.warning("TensorFlow not found — using stub predictor.")


# ─────────────────────────────────────────────────────────────────────────────
# STUB PREDICTOR (used when TF is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class _StubModel:
    """
    Deterministic stub that mimics the real model's predict() interface.
    Returns plausible values derived from input statistics so that the
    full API stack can be exercised without a trained checkpoint.
    """

    version = "stub-1.0.0"

    def predict(self, inputs: list[np.ndarray], verbose: int = 0) -> dict:
        X_num = inputs[0]  # (batch, T, 6)
        batch = X_num.shape[0]

        # Derive pseudo-predictions from feature means
        mean_score      = float(X_num[:, :, 0].mean(axis=1).mean())
        mean_engagement = float(X_num[:, :, 1].mean(axis=1).mean())
        mean_hint       = float(X_num[:, :, 2].mean(axis=1).mean())

        perf    = np.clip(mean_score * 100, 0, 100)
        mastery = np.clip(mean_engagement - mean_hint * 0.05, 0.05, 0.95)
        dropout = np.clip(1 - mean_engagement + mean_hint * 0.08, 0.02, 0.98)

        return {
            "performance_score": np.full((batch, 1), perf,    dtype=np.float32),
            "mastery_prob":      np.full((batch, 1), mastery, dtype=np.float32),
            "dropout_risk":      np.full((batch, 1), dropout, dtype=np.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Loads and holds:
      • the Keras LSTM model (or stub)
      • the fitted MinMaxScaler from training
      • a simple in-memory cache of recent predictions

    Lifecycle
    ---------
    Call `load()` once at application startup (lifespan handler).
    Call `predict()` from any route handler.
    """

    CHECKPOINT_PATH = Path("artifacts/checkpoints/best_model.keras")
    SCALER_PATH     = Path("artifacts/checkpoints/scaler.npy")

    RISK_THRESHOLDS = {"high": 0.65, "medium": 0.40, "low": 0.0}

    INTERVENTION_MAP = {
        (True,  True):  "ALERT_INSTRUCTOR",       # high dropout, low mastery
        (True,  False): "ALERT_INSTRUCTOR",
        (False, True):  "ASSIGN_REMEDIAL_MODULE",  # low perf, low mastery
        (False, False): "CONTINUE_STANDARD_PATH",
    }

    def __init__(self):
        self._model:       Optional[object]    = None
        self._scaler:      Optional[MinMaxScaler] = None
        self._loaded_at:   Optional[float]     = None
        self._version:     str                 = "unloaded"
        self._predict_count: int               = 0

    # ── Startup ──────────────────────────────────────────────────────────────

    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        """
        Load model weights and scaler from disk.
        Falls back to stub + fresh scaler if files are missing.
        """
        t0 = time.perf_counter()
        cp = checkpoint_path or self.CHECKPOINT_PATH

        if _TF_AVAILABLE and cp.exists():
            logger.info("Loading Keras model from %s …", cp)
            self._model = keras.models.load_model(str(cp))
            self._version = "1.0.0"
        else:
            logger.warning("Checkpoint not found at %s — using stub.", cp)
            self._model = _StubModel()
            self._version = _StubModel.version

        if self.SCALER_PATH.exists():
            scaler_params = np.load(str(self.SCALER_PATH), allow_pickle=True).item()
            self._scaler = MinMaxScaler()
            self._scaler.scale_    = scaler_params["scale_"]
            self._scaler.min_      = scaler_params["min_"]
            self._scaler.data_min_ = scaler_params["data_min_"]
            self._scaler.data_max_ = scaler_params["data_max_"]
            self._scaler.data_range_ = scaler_params["data_range_"]
            self._scaler.feature_names_in_ = None
            self._scaler.n_features_in_ = 6
            self._scaler.n_samples_seen_ = 1
        else:
            logger.warning("Scaler not found — using identity scaler.")
            self._scaler = MinMaxScaler()
            # Fit on plausible data ranges so the scaler is valid
            dummy = np.array([
                [0,  0.0, 0,  1,   0,  0],
                [100, 1.0, 10, 240, 50, 50],
            ], dtype=np.float32)
            self._scaler.fit(dummy)

        elapsed = time.perf_counter() - t0
        self._loaded_at = time.time()
        logger.info("Model registry ready (%.2fs) — version=%s", elapsed, self._version)

    # ── Core predict ─────────────────────────────────────────────────────────

    def predict(
        self,
        num_sequences: np.ndarray,   # (batch, T, 6)  raw unscaled
        cat_sequences: np.ndarray,   # (batch, T)     module_id ints
        return_attention: bool = False,
    ) -> list[dict]:
        """
        Run inference on a batch of learner sequences.

        Parameters
        ----------
        num_sequences : (B, T, 6) raw numerical features
        cat_sequences : (B, T)   module_id integer sequences
        return_attention : include attention weights in output

        Returns
        -------
        List of dicts, one per learner in the batch, each containing:
            performance_score, mastery_prob, dropout_risk,
            dropout_tier, intervention, [attention_weights]
        """
        if self._model is None:
            raise RuntimeError("ModelRegistry.load() must be called before predict()")

        B, T, F = num_sequences.shape

        # Scale numerical features
        flat    = num_sequences.reshape(-1, F)
        scaled  = self._scaler.transform(flat).reshape(B, T, F).astype(np.float32)

        preds = self._model.predict([scaled, cat_sequences], verbose=0)
        self._predict_count += B

        results = []
        for i in range(B):
            perf    = float(np.clip(preds["performance_score"][i, 0], 0, 100))
            mastery = float(np.clip(preds["mastery_prob"][i, 0],      0, 1))
            dropout = float(np.clip(preds["dropout_risk"][i, 0],      0, 1))

            tier = "low"
            for t_name, thresh in self.RISK_THRESHOLDS.items():
                if dropout >= thresh:
                    tier = t_name
                    break

            intervention = self._recommend(mastery, dropout, perf)

            result = {
                "performance_score": round(perf, 3),
                "mastery_prob":      round(mastery, 4),
                "dropout_risk":      round(dropout, 4),
                "dropout_tier":      tier,
                "intervention":      intervention,
                "model_version":     self._version,
            }

            if return_attention and hasattr(preds, "get") and "attention_weights" in preds:
                weights = preds["attention_weights"][i].tolist()
                result["attention_weights"] = {
                    "timesteps": list(range(T)),
                    "weights":   weights,
                }

            results.append(result)

        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._version

    @property
    def uptime(self) -> float:
        return time.time() - self._loaded_at if self._loaded_at else 0.0

    @property
    def predict_count(self) -> int:
        return self._predict_count

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _recommend(mastery: float, dropout: float, perf: float) -> str:
        if dropout > 0.65:
            return "ALERT_INSTRUCTOR"
        if mastery < 0.4 and perf < 60:
            return "ASSIGN_REMEDIAL_MODULE"
        if dropout > 0.40:
            return "SEND_MOTIVATIONAL_NUDGE"
        if mastery > 0.75:
            return "AWARD_BADGE"
        return "CONTINUE_STANDARD_PATH"


# ── Application-level singleton ───────────────────────────────────────────────
# Accessed via FastAPI's dependency injection (see main.py).

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """FastAPI dependency — returns the shared ModelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry