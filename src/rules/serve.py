"""
Lightweight loader for already-computed rule derivation artifacts (see
src/rules/rules.py for how these were generated via surrogate tree
distillation). Used by the API to serve rules instantly without
re-fitting the surrogate tree on every request.
"""

from pathlib import Path
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_rules_text() -> str:
    path = MODELS_DIR / "derived_rules.txt"
    if not path.exists():
        return "Rules not yet generated. Run: python -m src.rules.rules"
    return path.read_text(encoding="utf-8")


def load_depth_sensitivity() -> list:
    path = MODELS_DIR / "rule_depth_sensitivity.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
