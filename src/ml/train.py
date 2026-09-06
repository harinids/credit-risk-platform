"""
Phase 3 — ML training pipeline for the Home Credit Default Risk dataset.

Handles:
    - Relational feature matrix (from src.data.feature_engineering)
    - Categorical encoding
    - Class imbalance via scale_pos_weight (algorithm-level) — chosen over
      naive oversampling (SMOTE) because SMOTE on 200+ mixed numeric/categorical
      features with heavy NaNs tends to create unrealistic synthetic points;
      scale_pos_weight lets LightGBM's tree splits directly account for the
      11:1 imbalance without distorting the feature distributions.
    - Threshold tuning (not a fixed 0.5 cutoff) based on the validation set
    - Risk band assignment (Low / Medium / High) via PERCENTILES of the
      predicted probability distribution — not multiples of the F1
      threshold, which breaks down under scale_pos_weight rescaling (a
      fixed-multiple approach put 99.8% of applicants in Low/Medium and
      only 0.2% in High, which is useless for a bank).
    - Saves model + metadata (metrics, feature list, thresholds) for the API

Usage:
    python -m src.ml.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

from src.data.feature_engineering import build_feature_matrix

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ["sk_id_curr"]
TARGET_COL = "target"


def prepare_data():
    df = build_feature_matrix(is_train=True)
    df.columns = [c.lower() for c in df.columns]

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + ID_COLS)

    # LightGBM handles categoricals natively if dtype is 'category'
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    return X, y, cat_cols


def find_best_threshold(y_true, y_proba):
    """
    Sweep thresholds and pick the one maximizing F1 — better than a fixed
    0.5 cutoff, which is meaningless under 11:1 imbalance (0.5 would almost
    never fire on the minority class).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no matching threshold
    return thresholds[best_idx], f1_scores[best_idx]


def assign_risk_band(proba: float, low_cutoff: float, high_cutoff: float) -> str:
    if proba < low_cutoff:
        return "Low"
    elif proba < high_cutoff:
        return "Medium"
    return "High"


def main():
    print("Building feature matrix...")
    X, y, cat_cols = prepare_data()
    print(f"Feature matrix: {X.shape}, categorical columns: {len(cat_cols)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # scale_pos_weight = (# negative) / (# positive) — algorithm-level
    # imbalance correction, computed from the TRAIN split only to avoid
    # leaking validation-set class balance into training.
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Class imbalance -> scale_pos_weight = {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("Training LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_cols,
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    print(f"\nValidation AUC-ROC: {auc:.4f}")

    best_threshold, best_f1 = find_best_threshold(y_val, y_proba)
    print(f"Best threshold (F1-optimized): {best_threshold:.4f} (F1 = {best_f1:.4f})")

    y_pred = (y_proba >= best_threshold).astype(int)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print(f"\nAt tuned threshold ({best_threshold:.4f}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\nConfusion matrix:\n{cm}")
    print(f"\nClassification report:\n{classification_report(y_val, y_pred)}")

    # Risk bands: percentile-based on the validation set's predicted
    # probability distribution, NOT arbitrary multiples of the F1 threshold.
    # Business framing: bottom 60% of applicants by risk = Low, next 30% =
    # Medium (needs manual review), top 10% = High (auto-flag/decline).
    # This guarantees a usable, non-degenerate split regardless of how the
    # raw probabilities are scaled by scale_pos_weight.
    low_cutoff = float(np.percentile(y_proba, 60))
    high_cutoff = float(np.percentile(y_proba, 90))
    print(f"\nRisk bands (percentile-based): Low < {low_cutoff:.3f} | "
          f"Medium < {high_cutoff:.3f} | High >= {high_cutoff:.3f}")

    bands = pd.Series([assign_risk_band(p, low_cutoff, high_cutoff) for p in y_proba])
    band_counts = bands.value_counts()
    print(f"\nValidation set risk band distribution:\n{band_counts}")

    # Sanity check the bands actually separate risk: default rate should
    # rise sharply from Low -> Medium -> High. This is the real proof the
    # bands are doing their job (and a good chart for the presentation).
    band_default_rates = pd.DataFrame({"band": bands, "actual_target": y_val.values}) \
        .groupby("band")["actual_target"].mean().sort_values()
    print(f"\nActual default rate by risk band (should increase Low -> High):\n{band_default_rates}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\nTop 15 features by importance:\n{importance.head(15)}")

    # --- Save artifacts ---
    model_path = MODELS_DIR / "risk_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    metadata = {
        "auc_roc": float(auc),
        "best_threshold": float(best_threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "scale_pos_weight": float(scale_pos_weight),
        "risk_bands": {
            "low_cutoff": float(low_cutoff),
            "high_cutoff": float(high_cutoff),
            "method": "percentile_60_90",
        },
        "band_default_rates": band_default_rates.to_dict(),
        "feature_columns": X.columns.tolist(),
        "categorical_columns": cat_cols,
        "top_15_features": importance.head(15).to_dict(),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()