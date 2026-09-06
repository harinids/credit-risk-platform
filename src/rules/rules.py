"""
Phase 4 (cont.) — Rule derivation via surrogate model distillation.

Rather than hand-writing thresholds ("if debt-to-income > 0.6 then High
risk"), we fit a shallow decision tree on the BLACK-BOX MODEL'S PREDICTED
PROBABILITIES (not the raw labels). This distills LightGBM's learned
behavior into a small set of human-readable rules that approximate it,
with a measurable fidelity score showing how well the rules track the
real model — a defensible way to generate audit-friendly policy rules
instead of guessing thresholds manually.

We also run a depth-sensitivity analysis: fidelity vs. rule-count is a
real trade-off (deeper trees fit the model better but produce far more
rules than a business analyst can act on), and documenting that trade-off
explicitly is part of a defensible methodology — better than picking
max_depth=4 arbitrarily with no justification.

Usage:
    python -m src.rules.rules
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.ml.train import prepare_data

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SURROGATE_MAX_DEPTH = 4


def recreate_validation_split():
    X, y, cat_cols = prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_val, y_val, cat_cols


def encode_categoricals_for_tree(X: pd.DataFrame) -> pd.DataFrame:
    """
    sklearn's DecisionTreeRegressor can't take pandas 'category' dtype
    directly like LightGBM can — encode categoricals to integer codes
    just for the surrogate tree (fine, since we only need rule structure,
    not the categorical semantics LightGBM uses internally).
    """
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=["category"]).columns:
        X_encoded[col] = X_encoded[col].cat.codes
    return X_encoded


def fit_surrogate_tree(X: pd.DataFrame, model_proba: np.ndarray, max_depth: int = SURROGATE_MAX_DEPTH):
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=200, random_state=42)
    tree.fit(X, model_proba)
    return tree


def compute_fidelity(tree, X: pd.DataFrame, model_proba: np.ndarray) -> float:
    """
    Fidelity = how well the surrogate tree's predictions match the real
    model's predictions (R^2), NOT how well either matches ground truth.
    This is the correct metric for distillation quality.
    """
    tree_proba = tree.predict(X)
    return r2_score(model_proba, tree_proba)


def depth_sensitivity_analysis(X: pd.DataFrame, model_proba: np.ndarray, depths=(2, 3, 4, 5, 6, 8)):
    """
    Shows the interpretability/fidelity trade-off explicitly: deeper trees
    fit the real model better but produce more rules and are harder for a
    business analyst to act on. Documenting this trade-off (rather than
    picking max_depth=4 arbitrarily) is itself part of a defensible rule
    derivation methodology.
    """
    print(f"\n{'=' * 70}")
    print("DEPTH SENSITIVITY — fidelity vs. rule-count trade-off")
    print(f"{'=' * 70}")
    results = []
    for d in depths:
        t = DecisionTreeRegressor(max_depth=d, min_samples_leaf=200, random_state=42)
        t.fit(X, model_proba)
        fid = compute_fidelity(t, X, model_proba)
        n_leaves = t.get_n_leaves()
        results.append({"max_depth": d, "fidelity_r2": fid, "n_rules": n_leaves})
        print(f"  max_depth={d}: fidelity R^2 = {fid:.4f}, rules = {n_leaves}")
    return pd.DataFrame(results)


def extract_leaf_rules(tree, feature_names, X: pd.DataFrame, model_proba: np.ndarray):
    """
    Walks the tree and produces one human-readable rule per leaf, with:
      - the path of conditions leading to that leaf
      - the average predicted probability in that leaf (from the surrogate)
      - the risk band that probability maps to
      - how many validation applicants fall into that leaf (support)
    """
    tree_ = tree.tree_
    leaf_id = tree.apply(X)

    rules = []

    def recurse(node, path):
        if tree_.feature[node] == -2:  # leaf node
            mask = leaf_id == node
            support = int(mask.sum())
            if support == 0:
                return
            avg_proba = float(model_proba[mask].mean())
            rules.append({
                "conditions": list(path),
                "predicted_probability": avg_proba,
                "support": support,
                "support_pct": round(support / len(X) * 100, 2),
            })
            return

        feature = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]

        recurse(tree_.children_left[node], path + [f"{feature} <= {threshold:.3f}"])
        recurse(tree_.children_right[node], path + [f"{feature} > {threshold:.3f}"])

    recurse(0, [])
    return sorted(rules, key=lambda r: r["predicted_probability"], reverse=True)


def assign_band(proba, low_cutoff, high_cutoff):
    if proba < low_cutoff:
        return "Low"
    elif proba < high_cutoff:
        return "Medium"
    return "High"


def main():
    print("Loading model and metadata...")
    model = joblib.load(MODELS_DIR / "risk_model.joblib")
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    low_cutoff = metadata["risk_bands"]["low_cutoff"]
    high_cutoff = metadata["risk_bands"]["high_cutoff"]

    print("Recreating validation split...")
    X_val, y_val, cat_cols = recreate_validation_split()

    print("Getting model predictions (the surrogate learns to mimic these)...")
    model_proba = model.predict_proba(X_val)[:, 1]

    print("Encoding categoricals for surrogate tree...")
    X_encoded = encode_categoricals_for_tree(X_val)

    print(f"Fitting surrogate decision tree (max_depth={SURROGATE_MAX_DEPTH})...")
    tree = fit_surrogate_tree(X_encoded, model_proba)

    fidelity = compute_fidelity(tree, X_encoded, model_proba)
    print(f"\nSurrogate fidelity (R^2 vs real model): {fidelity:.4f}")
    print("(This measures how well the simple rules approximate LightGBM's "
          "behavior, not accuracy vs ground truth — a high value means the "
          "rules are a trustworthy summary of what the model actually does.)")

    sensitivity_df = depth_sensitivity_analysis(X_encoded, model_proba)
    sensitivity_df.to_csv(MODELS_DIR / "rule_depth_sensitivity.csv", index=False)
    print(f"\nDepth sensitivity table saved to {MODELS_DIR / 'rule_depth_sensitivity.csv'}")

    rules = extract_leaf_rules(tree, X_encoded.columns.tolist(), X_encoded, model_proba)

    print(f"\n{'=' * 70}")
    print(f"DERIVED BUSINESS RULES ({len(rules)} leaf rules, max_depth={SURROGATE_MAX_DEPTH})")
    print(f"{'=' * 70}")

    for i, rule in enumerate(rules, 1):
        band = assign_band(rule["predicted_probability"], low_cutoff, high_cutoff)
        conditions_str = " AND ".join(rule["conditions"])
        print(f"\nRule {i}: [{band} RISK]")
        print(f"  IF {conditions_str}")
        print(f"  THEN predicted default probability ≈ {rule['predicted_probability']:.1%}")
        print(f"  (covers {rule['support']:,} applicants, {rule['support_pct']}% of validation set)")

    # Save readable rule text + the raw tree structure + depth sensitivity
    # note for reference
    tree_text = export_text(tree, feature_names=X_encoded.columns.tolist(), max_depth=SURROGATE_MAX_DEPTH)
    rules_output_path = MODELS_DIR / "derived_rules.txt"
    with open(rules_output_path, "w") as f:
        f.write(f"Surrogate fidelity (R^2 vs real model): {fidelity:.4f}\n")
        f.write(f"Chosen max_depth: {SURROGATE_MAX_DEPTH} (see rule_depth_sensitivity.csv "
                f"for the full fidelity-vs-rule-count trade-off)\n\n")
        f.write("DEPTH SENSITIVITY (fidelity vs. number of rules)\n")
        f.write("-" * 70 + "\n")
        f.write(sensitivity_df.to_string(index=False))
        f.write("\n\n")
        f.write(f"DERIVED BUSINESS RULES ({len(rules)} leaf rules)\n")
        f.write("=" * 70 + "\n")
        for i, rule in enumerate(rules, 1):
            band = assign_band(rule["predicted_probability"], low_cutoff, high_cutoff)
            conditions_str = " AND ".join(rule["conditions"])
            f.write(f"\nRule {i}: [{band} RISK]\n")
            f.write(f"  IF {conditions_str}\n")
            f.write(f"  THEN predicted default probability ~ {rule['predicted_probability']:.1%}\n")
            f.write(f"  (covers {rule['support']:,} applicants, {rule['support_pct']}% of validation set)\n")
        f.write("\n\nRAW TREE STRUCTURE\n")
        f.write("=" * 70 + "\n")
        f.write(tree_text)

    print(f"\nRules saved to {rules_output_path}")

    # Save the surrogate tree itself in case the API wants to reuse it
    joblib.dump(tree, MODELS_DIR / "surrogate_tree.joblib")
    print(f"Surrogate tree saved to {MODELS_DIR / 'surrogate_tree.joblib'}")


if __name__ == "__main__":
    main()