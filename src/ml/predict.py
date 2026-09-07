"""
Inference module for the risk model - loads the model + relational feature
matrix once, exposes predict_for_id() and explain_for_id() for the API.

Design note: because features are derived from relational tables (bureau,
previous_application, etc. aggregated per sk_id_curr), inference isn't a
simple "pass in raw fields" endpoint - it looks up an existing applicant's
precomputed feature row by sk_id_curr from the training set. This matches
a realistic bank workflow (scoring existing loan applications on file)
without requiring the caller to supply every engineered feature by hand.

Model + feature matrix are loaded ONCE at first use (module-level cache)
so repeated API calls are fast; only the first call pays the feature-
engineering cost (~1-2 minutes given relational joins across 6 tables).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.feature_engineering import build_feature_matrix

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_model = None
_metadata = None
_feature_df = None
_X_lookup = None
_shap_explainer = None


def _load_everything():
    global _model, _metadata, _feature_df, _X_lookup, _shap_explainer

    if _model is not None:
        return

    print("[predict] Loading model + metadata...")
    _model = joblib.load(MODELS_DIR / "risk_model.joblib")
    with open(MODELS_DIR / "model_metadata.json") as f:
        _metadata = json.load(f)

    print("[predict] Building feature matrix (one-time cost)...")
    df = build_feature_matrix(is_train=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.set_index("sk_id_curr", drop=False)

    feature_cols = _metadata["feature_columns"]
    cat_cols = _metadata["categorical_columns"]

    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    globals()["_feature_df"] = df
    globals()["_X_lookup"] = X

    print("[predict] Building SHAP explainer...")
    globals()["_shap_explainer"] = shap.TreeExplainer(_model)
    print("[predict] Ready.")


def get_sample_ids(limit: int = 20) -> list:
    _load_everything()
    return [int(x) for x in _X_lookup.index[:limit].tolist()]


def assign_band(proba: float) -> str:
    low = _metadata["risk_bands"]["low_cutoff"]
    high = _metadata["risk_bands"]["high_cutoff"]
    if proba < low:
        return "Low"
    elif proba < high:
        return "Medium"
    return "High"


def predict_for_id(sk_id_curr: int) -> dict:
    _load_everything()
    if sk_id_curr not in _X_lookup.index:
        raise KeyError(f"sk_id_curr {sk_id_curr} not found in dataset")

    row = _X_lookup.loc[[sk_id_curr]]
    proba = float(_model.predict_proba(row)[0, 1])
    band = assign_band(proba)
    actual_target = _feature_df.loc[sk_id_curr, "target"]

    return {
        "sk_id_curr": int(sk_id_curr),
        "default_probability": round(proba, 4),
        "risk_band": band,
        "actual_outcome": "Defaulted" if actual_target == 1 else "Repaid",
    }


def explain_for_id(sk_id_curr: int, top_n: int = 5) -> dict:
    _load_everything()
    if sk_id_curr not in _X_lookup.index:
        raise KeyError(f"sk_id_curr {sk_id_curr} not found in dataset")

    row = _X_lookup.loc[[sk_id_curr]]
    proba = float(_model.predict_proba(row)[0, 1])
    band = assign_band(proba)

    shap_raw = _shap_explainer.shap_values(row)
    if isinstance(shap_raw, list):
        shap_values = shap_raw[1][0]
    elif np.ndim(shap_raw) == 3:
        shap_values = shap_raw[0, :, 1]
    else:
        shap_values = shap_raw[0]

    contributions = pd.DataFrame({
        "feature": row.columns,
        "shap_value": shap_values,
        "feature_value": row.iloc[0].values,
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    top = contributions.sort_values("abs_shap", ascending=False).head(top_n)

    factors = []
    for _, r in top.iterrows():
        factors.append({
            "feature": r["feature"],
            "value": str(r["feature_value"]),
            "impact": round(float(r["shap_value"]), 4),
            "direction": "increases risk" if r["shap_value"] > 0 else "decreases risk",
        })

    summary = f"This applicant is flagged {band} risk (predicted default probability: {proba:.1%}). "
    summary += "Main factors: " + "; ".join(
        f"{f['feature'].replace('_', ' ')} {f['direction']}" for f in factors[:3]
    ) + "."

    return {
        "sk_id_curr": int(sk_id_curr),
        "default_probability": round(proba, 4),
        "risk_band": band,
        "top_factors": factors,
        "explanation": summary,
    }
