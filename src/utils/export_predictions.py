"""
src/utils/export_predictions.py
================================
Export learner predictions to JSON format.

Usage:
    python -m src.utils.export_predictions --output predictions.json --data test
    python -m src.utils.export_predictions --output predictions.json --data all

Outputs:
    {
        "export_metadata": {
            "timestamp": "2026-03-26T20:30:59",
            "model_version": "1.0.0",
            "total_learners": 90,
            "split": "test"
        },
        "learners": [
            {
                "learner_id": "L001",
                "performance_score": 72.5,
                "mastery_prob": 0.87,
                "dropout_risk": 0.32,
                "dropout_tier": "low",
                "intervention": "AWARD_BADGE",
                "num_sessions": 10
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config import cfg
from api.model_registry import ModelRegistry
from data_pipeline.sequencer import Sequencer

logger = logging.getLogger("learnflow.export_predictions")

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def _resolve_tier(dropout_risk: float) -> str:
    """Determine dropout risk tier."""
    if dropout_risk >= 0.65:
        return "high"
    elif dropout_risk >= 0.40:
        return "medium"
    return "low"


def _resolve_intervention(performance: float, mastery: float, dropout_risk: float) -> str:
    """Determine recommended intervention."""
    if dropout_risk >= 0.65 and mastery < 0.40:
        return "ALERT_INSTRUCTOR"
    elif dropout_risk < 0.65 and mastery < 0.40:
        return "ASSIGN_REMEDIAL_MODULE"
    elif 0.40 <= dropout_risk < 0.65:
        return "SEND_MOTIVATIONAL_NUDGE"
    elif mastery >= 0.75:
        return "AWARD_BADGE"
    return "CONTINUE_STANDARD_PATH"


def export_predictions(
    output_path: Path,
    data_split: str = "test",
    checkpoint_path: Optional[Path] = None) -> dict:
    """
    Load model, run predictions on a data split, and export to JSON.

    Parameters
    ----------
    output_path : Path
        Where to save JSON file
    data_split : str
        Which data split to export: 'train', 'val', 'test', or 'all'
    checkpoint_path : Path, optional
        Path to model checkpoint (default: from config)

   """
    logger.info("=" * 54)
    logger.info("  LearnFlow · Prediction Export")
    logger.info("=" * 54)

    # ── Load model ───────────────────────────────────────────────────
    logger.info("[1/3] Loading model and scaler …")
    registry = ModelRegistry()
    if checkpoint_path is None:
        checkpoint_path = cfg.paths.checkpoints / cfg.model.checkpoint_name
    else:
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / cfg.model.checkpoint_name
    
    try:
        registry.load(checkpoint_path=checkpoint_path.parent)
        logger.info(f"Model loaded: v{registry.version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # ── Load sequences ───────────────────────────────────────────────
    logger.info(f"[2/3] Loading {data_split} data sequences …")
    seq = Sequencer()
    
    splits_to_load = []
    if data_split == "all":
        splits_to_load = ["train", "val", "test"]
    else:
        splits_to_load = [data_split]

    all_predictions = []
    total_learners = 0

    for split_name in splits_to_load:
        try:
            X_num, X_cat, y = seq.load(cfg.paths.sequences, split_name)
            num_samples = X_num.shape[0]
            logger.info(f"  {split_name}: {num_samples} learners")
            total_learners += num_samples

            # ── Run predictions ──────────────────────────────────────
            logger.info(f"  Running predictions on {split_name} set …")
            preds = registry.predict(
                num_sequences=X_num,
                cat_sequences=X_cat,
                return_attention=False,
            )

            # ── Collect results ──────────────────────────────────────
            for i in range(num_samples):
                perf_score = float(preds[i]["performance_score"])
                mastery_p = float(preds[i]["mastery_prob"])
                dropout_p = float(preds[i]["dropout_risk"])
                
                # Count non-zero sessions (actual length of sequence)
                num_sessions = int(np.count_nonzero(X_num[i, :, 0]))
                
                tier = _resolve_tier(dropout_p)
                intervention = _resolve_intervention(perf_score, mastery_p, dropout_p)

                all_predictions.append({
                    "learner_id": f"L_{split_name}_{i:04d}",
                    "split": split_name,
                    "performance_score": round(perf_score, 2),
                    "mastery_prob": round(mastery_p, 4),
                    "dropout_risk": round(dropout_p, 4),
                    "dropout_tier": tier,
                    "intervention": intervention,
                    "num_sessions": num_sessions,
                })

        except FileNotFoundError:
            logger.warning(f"  No data found for split '{split_name}'")
            continue

    # ── Export to JSON ───────────────────────────────────────────────
    logger.info(f"[3/3] Exporting {len(all_predictions)} predictions to JSON …")

    export_data = {
        "export_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_version": registry.version,
            "total_learners": len(all_predictions),
            "splits_included": splits_to_load,
            "checkpoint_path": str(checkpoint_path),
        },
        "learners": all_predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"✓ Exported to {output_path}")
    logger.info(f"  Total learners: {len(all_predictions)}")
    
    # Print summary statistics
    tiers = {}
    interventions = {}
    for pred in all_predictions:
        tiers[pred["dropout_tier"]] = tiers.get(pred["dropout_tier"], 0) + 1
        interventions[pred["intervention"]] = interventions.get(pred["intervention"], 0) + 1

    print("\n" + "=" * 54)
    print(f"  Dropout Risk Breakdown")
    print("=" * 54)
    for tier in ["high", "medium", "low"]:
        if tier in tiers:
            pct = 100 * tiers[tier] / len(all_predictions)
            print(f"  {tier.upper():8s}: {tiers[tier]:4d} ({pct:5.1f}%)")

    print(f"\n  Interventions")
    print("=" * 54)
    for iv_type in sorted(interventions.keys()):
        count = interventions[iv_type]
        pct = 100 * count / len(all_predictions)
        print(f"  {iv_type:32s}: {count:4d} ({pct:5.1f}%)")

    print("=" * 54)

    return {
        "output_path": str(output_path),
        "total_learners": len(all_predictions),
        "tiers": tiers,
        "interventions": interventions,
    }


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export learner predictions to JSON"
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default="artifacts/predictions.json",
        help="Output JSON file path"
    )
    p.add_argument(
        "--data", "-d",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which data split to export"
    )
    p.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (default: from config)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_predictions(
        output_path=Path(args.output),
        data_split=args.data,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )
