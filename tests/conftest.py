from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports like `from src.api.main import app` work during pytest collection.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
