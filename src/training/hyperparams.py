"""
src/training/hyperparams.py
============================
Optuna-based hyperparameter search for the LSTM model.

Run:
    python -m src.training.hyperparams --n-trials 50

Searches over:
    lstm_units_1, lstm_units_2, embed_dim, dropout_rate,
    l2_reg, learning_rate, batch_size
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from configs.config import cfg

logger = logging.getLogger("learnflow.hyperparams")


def objective(trial, splits: dict) -> float:
    """
    Optuna objective — minimise validation loss.
    Returns val_loss (lower is better).
    """
    try:
        import optuna
        import tensorflow as tf
        from keras._tf_keras import keras
    except ImportError as e:
        raise ImportError(f"optuna and tensorflow are required: {e}")

    from src.models.lstm import build_lstm_model

    lstm1    = trial.suggest_categorical("lstm_units_1",  [64, 128, 256])
    lstm2    = trial.suggest_categorical("lstm_units_2",  [32, 64, 128])
    embed    = trial.suggest_categorical("embed_dim",     [4, 8, 16])
    dropout  = trial.suggest_float("dropout_rate",        0.1, 0.5)
    l2       = trial.suggest_float("l2_reg",              1e-5, 1e-3, log=True)
    lr       = trial.suggest_float("learning_rate",       1e-4, 1e-2, log=True)
    batch    = trial.suggest_categorical("batch_size",    [16, 32, 64])

    model = build_lstm_model(
        seq_len=cfg.data.seq_len, num_features=cfg.data.n_features,
        num_modules=cfg.data.n_modules,
        embed_dim=embed, lstm_units_1=lstm1, lstm_units_2=lstm2,
        dropout_rate=dropout, l2_reg=l2,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={
            "performance_score": keras.losses.MeanSquaredError(),
            "mastery_prob":      keras.losses.BinaryCrossentropy(),
            "dropout_risk":      keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "performance_score": 0.30,
            "mastery_prob":      0.35,
            "dropout_risk":      0.35,
        },
    )

    X_num_tr, X_cat_tr, y_tr   = splits["train"]
    X_num_val, X_cat_val, y_val = splits["val"]

    history = model.fit(
        [X_num_tr, X_cat_tr],
        {"performance_score": y_tr[:, 0],
         "mastery_prob":      y_tr[:, 1],
         "dropout_risk":      y_tr[:, 2]},
        validation_data=(
            [X_num_val, X_cat_val],
            {"performance_score": y_val[:, 0],
             "mastery_prob":      y_val[:, 1],
             "dropout_risk":      y_val[:, 2]},
        ),
        epochs=20, batch_size=batch, verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        ],
    )
    return min(history.history["val_loss"])


def run_search(n_trials: int = 30) -> dict:
    """Run the Optuna study and return best hyperparameters."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    from src.training.trainer import load_data
    splits = load_data()

    study = optuna.create_study(direction="minimize",
                                study_name="learnflow_lstm")
    study.optimize(lambda t: objective(t, splits), n_trials=n_trials,
                   show_progress_bar=True)

    best = study.best_params
    logger.info("Best params: %s", json.dumps(best, indent=2))
    logger.info("Best val_loss: %.5f", study.best_value)

    out = cfg.paths.checkpoints / "best_hyperparams.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"best_params": best, "best_val_loss": study.best_value}, f, indent=2)
    logger.info("Saved best hyperparams to %s", out)
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    args = p.parse_args()
    run_search(args.n_trials)