"""
Phase 4 — Explainability for the credit risk model.

Provides:
    - Global explainability: which features matter most across all
      applicants (SHAP summary/bar plot)
    - Local explainability: for one applicant, which specific features
      pushed their score up or down, and by how much
    - Plain-English explanation generation, grounded strictly in the
      SHAP values themselves (no free-form LLM generation here — this
      avoids hallucination, since every sentence is derived directly
      from a computed number, not invented). LLM polish of the phrasing
      gets layered on later when this is wired into the API.

Usage:
    python -m src.explainability.explain
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.ml.train import prepare_data

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 1000  # SHAP on the full 61k validation set is slow; a sample is enough for insight


def load_artifacts():
    model = joblib.load(MODELS_DIR / "risk_model.joblib")
    return model


def recreate_validation_split():
    """
    Reruns the exact same feature engineering + split as train.py
    (same random_state=42) so we get back an identical X_val/y_val
    without needing to have saved it separately.
    """
    X, y, cat_cols = prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_val, y_val


def extract_class1_shap(shap_raw, explainer):
    """
    Handles the different shapes shap.TreeExplainer can return depending
    on version: list[neg, pos], (n, features), or (n, features, classes).
    Always returns the SHAP values for the positive (default) class.
    """
    if isinstance(shap_raw, list):
        shap_values = shap_raw[1]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    elif shap_raw.ndim == 3:
        shap_values = shap_raw[:, :, 1]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        shap_values = shap_raw
        base_value = explainer.expected_value
    return shap_values, base_value


def explain_instance(row_idx, X_sample, shap_values, top_n=5):
    """
    Returns the top_n features that most influenced this specific
    prediction, with their SHAP values and direction (increases/decreases risk).
    """
    row_shap = shap_values[row_idx]
    row_data = X_sample.iloc[row_idx]

    contributions = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_value": row_shap,
        "feature_value": row_data.values,
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    top = contributions.sort_values("abs_shap", ascending=False).head(top_n)

    return top


def generate_plain_english_explanation(top_contributions: pd.DataFrame, proba: float, band: str) -> str:
    """
    Template-based explanation, strictly grounded in the SHAP values
    passed in. No invented content — every clause maps to an actual
    row in top_contributions.
    """
    lines = [f"This applicant was flagged **{band} risk** (predicted default probability: {proba:.1%})."]
    lines.append("The main factors behind this assessment:")

    for _, row in top_contributions.iterrows():
        direction = "increased" if row["shap_value"] > 0 else "decreased"
        feature_name = row["feature"].replace("_", " ")
        lines.append(
            f"  - {feature_name} = {row['feature_value']} → {direction} risk "
            f"(impact: {row['shap_value']:+.4f})"
        )

    return "\n".join(lines)


def main():
    print("Loading model...")
    model = load_artifacts()

    print("Recreating validation split (must match train.py exactly)...")
    X_val, y_val = recreate_validation_split()

    print(f"Sampling {SAMPLE_SIZE} rows for SHAP computation (full set would be slow)...")
    sample_idx = X_val.sample(n=min(SAMPLE_SIZE, len(X_val)), random_state=42).index
    X_sample = X_val.loc[sample_idx].reset_index(drop=True)
    y_sample = y_val.loc[sample_idx].reset_index(drop=True)

    y_proba_sample = model.predict_proba(X_sample)[:, 1]

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_sample)
    shap_values, base_value = extract_class1_shap(shap_raw, explainer)
    print(f"SHAP values computed: shape {shap_values.shape}, base value = {base_value:.4f}")

    # --- Global explainability ---
    print("\n" + "=" * 70)
    print("GLOBAL EXPLAINABILITY — mean |SHAP value| per feature")
    print("=" * 70)
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)
    print(mean_abs_shap.head(15))

    plt.figure(figsize=(9, 6))
    mean_abs_shap.head(15).sort_values().plot(kind="barh", color="#4C72B0")
    plt.title("Global Feature Importance (mean |SHAP value|)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_shap_global_importance.png", dpi=120)
    plt.close()
    print(f"\nGlobal SHAP chart saved to {OUTPUT_DIR / '05_shap_global_importance.png'}")

    # --- Local explainability: 3 illustrative examples ---
    print("\n" + "=" * 70)
    print("LOCAL EXPLAINABILITY — 3 example applicants")
    print("=" * 70)

    highest_idx = int(np.argmax(y_proba_sample))
    lowest_idx = int(np.argmin(y_proba_sample))
    # borderline: closest to the median probability
    median_proba = np.median(y_proba_sample)
    borderline_idx = int(np.argmin(np.abs(y_proba_sample - median_proba)))

    examples = {
        "HIGHEST RISK": highest_idx,
        "LOWEST RISK": lowest_idx,
        "BORDERLINE / MEDIAN": borderline_idx,
    }

    for label, idx in examples.items():
        proba = y_proba_sample[idx]
        band = "High" if proba >= 0.702 else "Medium" if proba >= 0.415 else "Low"
        top_contrib = explain_instance(idx, X_sample, shap_values, top_n=5)
        explanation = generate_plain_english_explanation(top_contrib, proba, band)

        print(f"\n--- {label} (row {idx}, actual outcome: "
              f"{'Defaulted' if y_sample.iloc[idx] == 1 else 'Repaid'}) ---")
        print(explanation)

    # Save a waterfall plot for the highest-risk example
    try:
        exp_obj = shap.Explanation(
            values=shap_values[highest_idx],
            base_values=base_value,
            data=X_sample.iloc[highest_idx].values,
            feature_names=X_sample.columns.tolist(),
        )
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(exp_obj, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_shap_waterfall_highest_risk.png", dpi=120)
        plt.close()
        print(f"\nWaterfall plot saved to {OUTPUT_DIR / '06_shap_waterfall_highest_risk.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not render waterfall plot (non-fatal): {exc}")


if __name__ == "__main__":
    main()