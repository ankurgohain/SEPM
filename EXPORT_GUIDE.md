# Exporting Learner Predictions to JSON

## Quick Start

Export predictions from your test set:

```bash
python -m src.utils.export_predictions --output predictions.json
```

Or specify a different data split:

```bash
# Export all splits (train + val + test)
python -m src.utils.export_predictions --output all_predictions.json --data all

# Export only validation set
python -m src.utils.export_predictions --output val_predictions.json --data val

# Export training set
python -m src.utils.export_predictions --output train_predictions.json --data train
```

## Output Format

The JSON file contains:

```json
{
  "export_metadata": {
    "timestamp": "2026-03-26T20:30:59.123456",
    "model_version": "1.0.0",
    "total_learners": 90,
    "splits_included": ["test"],
    "checkpoint_path": "artifacts/checkpoints"
  },
  "learners": [
    {
      "learner_id": "L_test_0001",
      "split": "test",
      "performance_score": 72.5,
      "mastery_prob": 0.8734,
      "dropout_risk": 0.3211,
      "dropout_tier": "low",
      "intervention": "AWARD_BADGE",
      "num_sessions": 10
    },
    {
      "learner_id": "L_test_0002",
      "split": "test",
      "performance_score": 45.3,
      "mastery_prob": 0.2145,
      "dropout_risk": 0.7123,
      "dropout_tier": "high",
      "intervention": "ALERT_INSTRUCTOR",
      "num_sessions": 8
    },
    ...
  ]
}
```

## Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `learner_id` | string | Unique identifier (format: L_{split}_{index}) |
| `split` | string | Data split (train/val/test) |
| `performance_score` | float (0-100) | Predicted quiz performance |
| `mastery_prob` | float (0-1) | Probability of concept mastery |
| `dropout_risk` | float (0-1) | Probability of dropout |
| `dropout_tier` | string | Risk category (low/medium/high) |
| `intervention` | string | Recommended action |
| `num_sessions` | int | Number of active sessions in sequence |

## Intervention Types

- **ALERT_INSTRUCTOR** — Critical: High dropout risk + low mastery. Instructor follow-up needed.
- **ASSIGN_REMEDIAL_MODULE** — Low mastery. Assign extra practice content.
- **SEND_MOTIVATIONAL_NUDGE** — Medium dropout risk. Send encouragement message.
- **AWARD_BADGE** — High mastery. Reward and unlock next module.
- **CONTINUE_STANDARD_PATH** — On track. No intervention needed.

## Dropout Tier Breakdown

- **HIGH** — dropout_risk ≥ 0.65 (critical)
- **MEDIUM** — 0.40 ≤ dropout_risk < 0.65 (caution)
- **LOW** — dropout_risk < 0.40 (healthy)

## Command-Line Options

```bash
usage: python -m src.utils.export_predictions [-h] [--output OUTPUT] 
                                               [--data {train,val,test,all}] 
                                               [--checkpoint CHECKPOINT]

Optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output JSON file path (default: artifacts/predictions.json)
  --data {train,val,test,all}, -d {train,val,test,all}
                        Which data split to export (default: test)
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        Path to model checkpoint (default: from config)
```

## Example Workflows

### Use Case 1: Monitor Test Set Performance

```bash
python -m src.utils.export_predictions --output test_results.json --data test
```

Then analyze in Python:

```python
import json
with open('test_results.json') as f:
    data = json.load(f)

# Find high-risk learners
high_risk = [l for l in data['learners'] if l['dropout_tier'] == 'high']
print(f"High-risk learners: {len(high_risk)}")

# Calculate average mastery
avg_mastery = sum(l['mastery_prob'] for l in data['learners']) / len(data['learners'])
print(f"Average mastery: {avg_mastery:.2%}")
```

### Use Case 2: Export All Data for Dashboard

```bash
python -m src.utils.export_predictions --output all_data.json --data all
```

Then use in your frontend dashboard to display learner cards.

### Use Case 3: Create Intervention Report

```bash
python -m src.utils.export_predictions --output interventions.json --data test
```

The console output will show a breakdown:

```
══════════════════════════════════════════════════════
  Dropout Risk Breakdown
══════════════════════════════════════════════════════
  HIGH    :   12 (  13.3%)
  MEDIUM  :   23 (  25.6%)
  LOW     :   55 (  61.1%)

  Interventions
══════════════════════════════════════════════════════
  ALERT_INSTRUCTOR         :   12 (  13.3%)
  ASSIGN_REMEDIAL_MODULE   :    8 (   8.9%)
  SEND_MOTIVATIONAL_NUDGE  :   15 (  16.7%)
  AWARD_BADGE              :   42 (  46.7%)
  CONTINUE_STANDARD_PATH   :   13 (  14.4%)
══════════════════════════════════════════════════════
```

## Notes

- Learner IDs in the export are synthetic (format: `L_{split}_{index}`)
- If you have custom learner IDs, you'll need to map them manually
- The export includes only the latest checkpoint — older models require specifying `--checkpoint`
- All float values are rounded for JSON readability (mastery/dropout to 4 decimals, performance to 2 decimals)
