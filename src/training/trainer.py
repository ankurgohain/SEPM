"""
src/training/trainer.py
========================
Executable training script for the LearnFlow LSTM model.

Run from project root:
    python -m src.training.trainer
    python -m src.training.trainer --epochs 80 --batch-size 64

The trainer:
  1. Loads (or generates) data via the data pipeline
  2. Builds and compiles the LSTM model
  3. Runs training with early stopping + LR decay
  4. Evaluates on the held-out test set
  5. Saves the best checkpoint + scaler
  6. Optionally logs everything to MLflow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# ── Try TF import (required for actual training) ────────────────────────────
try:
    import tensorflow as tf
    from keras._tf_keras import keras
    from keras import callbacks as tf_callbacks
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from configs.config import cfg
from src.data_pipeline.ingestion import SyntheticIngester, CSVIngester
from src.data_pipeline.cleaner import EventCleaner
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.sequencer import Sequencer

logger = logging.getLogger("learnflow.trainer")

logging.basicConfig(
    level=cfg.api.log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER (imported from lstm_model to keep DRY)
# ─────────────────────────────────────────────────────────────────────────────

def _build_and_compile(mc=cfg.model, tc=cfg.training):
    """Build + compile the LSTM model using config values."""
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for training.")

    # Import here to avoid circular dependency
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lstm_model import build_lstm_model

    model = build_lstm_model(
        seq_len         = cfg.data.seq_len,
        num_features    = cfg.data.n_features,
        num_modules     = cfg.data.n_modules,
        embed_dim       = mc.embed_dim,
        lstm_units_1    = mc.lstm_units_1,
        lstm_units_2    = mc.lstm_units_2,
        attention_units = mc.attention_units,
        dropout_rate    = mc.dropout_rate,
        l2_reg          = mc.l2_reg,
    )

    reg = keras.regularizers.l2(mc.l2_reg)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=tc.learning_rate,
            clipnorm=tc.clipnorm,
        ),
        loss={
            "performance_score": keras.losses.MeanSquaredError(),
            "mastery_prob":      keras.losses.BinaryCrossentropy(),
            "dropout_risk":      keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "performance_score": mc.loss_weight_performance,
            "mastery_prob":      mc.loss_weight_mastery,
            "dropout_risk":      mc.loss_weight_dropout,
        },
        metrics={
            "performance_score": [keras.metrics.RootMeanSquaredError(name="rmse")],
            "mastery_prob":      [keras.metrics.AUC(name="auc")],
            "dropout_risk":      [keras.metrics.AUC(name="auc")],
        },
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def _get_callbacks(checkpoint_path: Path, tc=cfg.training):
    if not _TF_AVAILABLE:
        return []

    return [
        tf_callbacks.EarlyStopping(
            monitor             = tc.es_monitor,
            patience            = tc.es_patience,
            restore_best_weights= tc.restore_best,
            verbose             = 1,
        ),
        tf_callbacks.ReduceLROnPlateau(
            monitor  = tc.es_monitor,
            factor   = tc.lr_factor,
            patience = tc.lr_patience,
            min_lr   = tc.lr_min,
            verbose  = 1,
        ),
        tf_callbacks.ModelCheckpoint(
            filepath       = str(checkpoint_path),
            monitor        = tc.es_monitor,
            save_best_only = True,
            verbose        = 0,
        ),
        tf_callbacks.CSVLogger(
            str(checkpoint_path.parent / "training_log.csv"),
            append=True,
        ),
        tf_callbacks.TensorBoard(
            log_dir    = str(cfg.paths.logs / "tensorboard"),
            histogram_freq = 0,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    csv_path: Path | None = None,
    n_synthetic: int = None,
) -> tuple:
    """
    Load and pipeline data.

    Priority:
      1. CSV file if csv_path is provided
      2. Pre-saved tensors in cfg.paths.sequences
      3. Synthetic data generation
    """
    seq_dir = cfg.paths.sequences

    # Try loading pre-built tensors
    if (seq_dir / "train" / "X_num.npy").exists() and csv_path is None:
        logger.info("Loading pre-built tensors from %s", seq_dir)
        seq = Sequencer()
        train = seq.load(seq_dir, "train")
        val   = seq.load(seq_dir, "val")
        test  = seq.load(seq_dir, "test")
        return {"train": train, "val": val, "test": test}

    # Build from CSV or synthetic
    if csv_path and Path(csv_path).exists():
        logger.info("Ingesting CSV: %s", csv_path)
        ingester = CSVIngester(csv_path)
    else:
        logger.info("Generating synthetic data (n=%d)", n_synthetic or cfg.data.n_synthetic_learners)
        ingester = SyntheticIngester(n_learners=n_synthetic or cfg.data.n_synthetic_learners)

    cleaner = EventCleaner()
    fe      = FeatureEngineer()

    for batch in ingester.ingest():
        clean_batch, report = cleaner.clean(batch)
        fe.update(clean_batch)
        logger.info("Processed batch  %s", report)

    X_num, X_cat, y, learner_ids, scaler = fe.build_tensors(fit_scaler=True)

    seq = Sequencer()
    splits = seq.split(X_num, X_cat, y)

    # Save tensors and scaler for fast reloads
    cfg.paths.sequences.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)

    for split_name, (Xn, Xc, yl) in splits.items():
        seq.save(cfg.paths.sequences, Xn, Xc, yl, split=split_name)

    scaler_path = cfg.paths.checkpoints / cfg.model.scaler_name
    fe.save_scaler(scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_num, X_cat, y_true, split_name="Test") -> dict:
    """Run model.predict and compute per-head metrics."""
    from sklearn.metrics import mean_squared_error, roc_auc_score, classification_report

    preds    = model.predict([X_num, X_cat], verbose=0)
    perf_p   = preds["performance_score"].squeeze()
    mastery_p = preds["mastery_prob"].squeeze()
    dropout_p = preds["dropout_risk"].squeeze()

    perf_t    = y_true[:, 0]
    mastery_t = y_true[:, 1].astype(int)
    dropout_t = y_true[:, 2].astype(int)

    rmse = float(np.sqrt(mean_squared_error(perf_t, perf_p)))

    try:
        mastery_auc = float(roc_auc_score(mastery_t, mastery_p))
        dropout_auc = float(roc_auc_score(dropout_t, dropout_p))
    except ValueError:
        mastery_auc = dropout_auc = float("nan")

    sep = "─" * 54
    print(f"\n{'═'*54}")
    print(f"  {split_name} Evaluation")
    print(f"{'═'*54}")
    print(f"\n  Performance Score → RMSE : {rmse:.3f}")
    print(f"  Mastery Prob      → AUC  : {mastery_auc:.4f}")
    print(f"  Dropout Risk      → AUC  : {dropout_auc:.4f}")
    print(f"\n  Mastery classification report:")
    print(classification_report(mastery_t, (mastery_p >= 0.5).astype(int),
                                target_names=["Not Mastered", "Mastered"], indent=4))
    print(f"\n  Dropout risk classification report:")
    print(classification_report(dropout_t, (dropout_p >= 0.5).astype(int),
                                target_names=["Retained", "At Risk"], indent=4))
    print("═" * 54)

    return {
        "rmse_performance": rmse,
        "auc_mastery":      mastery_auc,
        "auc_dropout":      dropout_auc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW LOGGING (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _try_mlflow_log(params: dict, metrics: dict, checkpoint_path: Path) -> None:
    try:
        import mlflow
        mlflow.set_experiment(cfg.training.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if cfg.training.log_artifacts and checkpoint_path.exists():
                mlflow.log_artifact(str(checkpoint_path))
            logger.info("MLflow: run logged to experiment '%s'", cfg.training.mlflow_experiment)
    except ImportError:
        logger.debug("mlflow not installed — skipping experiment tracking")
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs:      int  = None,
    batch_size:  int  = None,
    csv_path:    Path | None = None,
    n_synthetic: int  = None,
    dry_run:     bool = False,
) -> dict:
    """
    Full end-to-end training pipeline.

    Parameters
    ----------
    epochs      : override cfg.training.epochs
    batch_size  : override cfg.training.batch_size
    csv_path    : path to real CSV data (uses synthetic if None)
    n_synthetic : number of synthetic learners (if no CSV)
    dry_run     : build data + model but skip fit() (for CI)

    Returns
    -------
    dict of test metrics
    """
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for training.\n"
            "Install it with: pip install tensorflow>=2.13"
        )

    tc = cfg.training
    mc = cfg.model

    np.random.seed(tc.seed)
    tf.random.set_seed(tc.seed)

    epochs_     = epochs     or tc.epochs
    batch_size_ = batch_size or tc.batch_size

    t0 = time.perf_counter()
    logger.info("═" * 54)
    logger.info("  LearnFlow · Training Pipeline")
    logger.info("═" * 54)

    # ── 1. Load data ─────────────────────────────────────────────────
    logger.info("[1/5] Loading data …")
    splits = load_data(csv_path=csv_path, n_synthetic=n_synthetic)

    X_num_tr, X_cat_tr, y_tr   = splits["train"]
    X_num_val, X_cat_val, y_val = splits["val"]
    X_num_te, X_cat_te, y_te   = splits["test"]

    train_targets = {
        "performance_score": y_tr[:, 0],
        "mastery_prob":      y_tr[:, 1],
        "dropout_risk":      y_tr[:, 2],
    }
    val_targets = {
        "performance_score": y_val[:, 0],
        "mastery_prob":      y_val[:, 1],
        "dropout_risk":      y_val[:, 2],
    }

    logger.info(
        "Data loaded  train=%d  val=%d  test=%d",
        len(X_num_tr), len(X_num_val), len(X_num_te),
    )

    # ── 2. Build model ───────────────────────────────────────────────
    logger.info("[2/5] Building model …")
    model = _build_and_compile()
    model.summary(line_length=72, print_fn=logger.info)

    if dry_run:
        logger.info("Dry run: skipping fit()")
        return {}

    # ── 3. Setup callbacks & checkpoint path ─────────────────────────
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.paths.checkpoints / mc.checkpoint_name
    cbs = _get_callbacks(checkpoint_path)

    # ── 4. Train ─────────────────────────────────────────────────────
    logger.info("[3/5] Training for up to %d epochs (batch=%d) …", epochs_, batch_size_)
    history = model.fit(
        x                = [X_num_tr, X_cat_tr],
        y                = train_targets,
        validation_data  = ([X_num_val, X_cat_val], val_targets),
        epochs           = epochs_,
        batch_size       = batch_size_,
        callbacks        = cbs,
        verbose          = 1,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────
    logger.info("[4/5] Evaluating on test set …")
    metrics = evaluate(model, X_num_te, X_cat_te, y_te, split_name="Test")

    # ── 6. Save checkpoint summary ───────────────────────────────────
    logger.info("[5/5] Saving run summary …")
    elapsed = time.perf_counter() - t0

    run_params = {
        "epochs_trained":  len(history.history["loss"]),
        "epochs_max":      epochs_,
        "batch_size":      batch_size_,
        "learning_rate":   tc.learning_rate,
        "lstm_units_1":    mc.lstm_units_1,
        "lstm_units_2":    mc.lstm_units_2,
        "dropout_rate":    mc.dropout_rate,
        "model_version":   mc.version,
        "elapsed_secs":    round(elapsed, 1),
    }

    summary_path = cfg.paths.checkpoints / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({**run_params, **metrics}, f, indent=2)
    logger.info("Run summary saved to %s", summary_path)

    # Optional MLflow
    _try_mlflow_log(run_params, metrics, checkpoint_path)

    logger.info("Training complete in %.1fs", elapsed)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="LearnFlow LSTM Trainer")
    p.add_argument("--epochs",       type=int,   default=None, help="Max training epochs")
    p.add_argument("--batch-size",   type=int,   default=None, help="Mini-batch size")
    p.add_argument("--csv",          type=str,   default=None, help="Path to CSV data file")
    p.add_argument("--n-synthetic",  type=int,   default=None, help="Number of synthetic learners")
    p.add_argument("--dry-run",      action="store_true",       help="Skip model.fit()")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    metrics = train(
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        csv_path    = Path(args.csv) if args.csv else None,
        n_synthetic = args.n_synthetic,
        dry_run     = args.dry_run,
    )
    print("\nFinal test metrics:", json.dumps(metrics, indent=2))