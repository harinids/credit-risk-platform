"""
Phase 2 — Relational feature engineering for the Home Credit Default Risk dataset.

Most submissions will only use application_train.csv. This module builds
aggregated features from the auxiliary tables (bureau, previous applications,
payment history) which carry most of the real predictive signal — e.g. past
default history, late payment patterns, credit utilization trends.

Usage:
    from src.data.feature_engineering import build_feature_matrix
    df = build_feature_matrix(is_train=True)
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _read_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known sentinel/anomaly values before any feature engineering."""
    df = df.copy()
    # DAYS_EMPLOYED sentinel (365243 = "not employed") -> NaN + explicit flag
    if "days_employed" in df.columns:
        df["days_employed_anomaly"] = (df["days_employed"] == 365243).astype(int)
        df.loc[df["days_employed"] == 365243, "days_employed"] = np.nan

    # Basic derived ratios (kept even though EDA showed weak standalone signal —
    # the model can still find interactions the univariate EDA couldn't)
    df["age_years"] = -df["days_birth"] / 365
    df["employed_years"] = -df["days_employed"] / 365
    df["credit_income_ratio"] = df["amt_credit"] / df["amt_income_total"]
    df["annuity_income_ratio"] = df["amt_annuity"] / df["amt_income_total"]
    df["credit_term_years"] = df["amt_credit"] / df["amt_annuity"]
    df["goods_price_credit_ratio"] = df["amt_goods_price"] / df["amt_credit"]

    return df


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    One row per sk_id_bureau in bureau_balance -> aggregate to sk_id_bureau,
    then aggregate bureau (per sk_id_curr) with those stats folded in.
    """
    # bureau_balance: status codes -> count of "days past due" months
    bb_agg = bureau_balance.groupby("sk_id_bureau").agg(
        bb_months_count=("months_balance", "count"),
        bb_dpd_count=("status", lambda s: s.isin(["1", "2", "3", "4", "5"]).sum()),
    ).reset_index()

    bureau = bureau.merge(bb_agg, on="sk_id_bureau", how="left")

    agg = bureau.groupby("sk_id_curr").agg(
        bureau_count=("sk_id_bureau", "count"),
        bureau_active_count=("credit_active", lambda s: (s == "Active").sum()),
        bureau_overdue_count=("credit_day_overdue", lambda s: (s > 0).sum()),
        bureau_max_overdue_days=("credit_day_overdue", "max"),
        bureau_total_debt=("amt_credit_sum_debt", "sum"),
        bureau_total_credit=("amt_credit_sum", "sum"),
        bureau_avg_credit=("amt_credit_sum", "mean"),
        bureau_bb_dpd_total=("bb_dpd_count", "sum"),
    ).reset_index()

    agg["bureau_debt_credit_ratio"] = (
        agg["bureau_total_debt"] / agg["bureau_total_credit"].replace(0, np.nan)
    )
    return agg


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    agg = prev.groupby("sk_id_curr").agg(
        prev_app_count=("sk_id_prev", "count"),
        prev_approved_count=("name_contract_status", lambda s: (s == "Approved").sum()),
        prev_refused_count=("name_contract_status", lambda s: (s == "Refused").sum()),
        prev_avg_credit=("amt_credit", "mean"),
        prev_avg_annuity=("amt_annuity", "mean"),
        prev_avg_application=("amt_application", "mean"),
    ).reset_index()

    agg["prev_refusal_rate"] = agg["prev_refused_count"] / agg["prev_app_count"].replace(0, np.nan)
    return agg


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    agg = pos.groupby("sk_id_curr").agg(
        pos_months_count=("months_balance", "count"),
        pos_max_dpd=("sk_dpd", "max"),
        pos_avg_dpd=("sk_dpd", "mean"),
        pos_completed_count=("name_contract_status", lambda s: (s == "Completed").sum()),
    ).reset_index()
    return agg


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    # Positive = paid late, negative/zero = on-time or early
    inst["days_late"] = inst["days_entry_payment"] - inst["days_instalment"]
    inst["payment_ratio"] = inst["amt_payment"] / inst["amt_instalment"].replace(0, np.nan)
    inst["underpaid"] = (inst["amt_payment"] < inst["amt_instalment"]).astype(int)

    agg = inst.groupby("sk_id_curr").agg(
        install_count=("sk_id_prev", "count"),
        install_avg_days_late=("days_late", "mean"),
        install_max_days_late=("days_late", "max"),
        install_late_count=("days_late", lambda s: (s > 0).sum()),
        install_underpaid_count=("underpaid", "sum"),
        install_avg_payment_ratio=("payment_ratio", "mean"),
    ).reset_index()

    agg["install_late_rate"] = agg["install_late_count"] / agg["install_count"].replace(0, np.nan)
    return agg


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    agg = cc.groupby("sk_id_curr").agg(
        cc_months_count=("months_balance", "count"),
        cc_avg_balance=("amt_balance", "mean"),
        cc_max_balance=("amt_balance", "max"),
        cc_avg_credit_limit=("amt_credit_limit_actual", "mean"),
        cc_max_dpd=("sk_dpd", "max"),
        cc_avg_drawings=("amt_drawings_current", "mean"),
    ).reset_index()

    agg["cc_utilization"] = (
        agg["cc_avg_balance"] / agg["cc_avg_credit_limit"].replace(0, np.nan)
    )
    return agg


def build_feature_matrix(is_train: bool = True) -> pd.DataFrame:
    """
    Loads application_{train|test}.csv plus every auxiliary table, builds
    aggregated relational features, and merges everything into one wide
    feature matrix keyed on sk_id_curr.
    """
    app_file = "application_train.csv" if is_train else "application_test.csv"
    app = _read_csv(app_file)
    app = clean_application(app)

    print(f"Loaded {app_file}: {app.shape}")

    bureau = _read_csv("bureau.csv")
    bureau_balance = _read_csv("bureau_balance.csv")
    bureau_agg = aggregate_bureau(bureau, bureau_balance)
    app = app.merge(bureau_agg, on="sk_id_curr", how="left")
    print(f"After bureau merge: {app.shape}")

    prev = _read_csv("previous_application.csv")
    prev_agg = aggregate_previous_application(prev)
    app = app.merge(prev_agg, on="sk_id_curr", how="left")
    print(f"After previous_application merge: {app.shape}")

    pos = _read_csv("POS_CASH_balance.csv")
    pos_agg = aggregate_pos_cash(pos)
    app = app.merge(pos_agg, on="sk_id_curr", how="left")
    print(f"After pos_cash merge: {app.shape}")

    inst = _read_csv("installments_payments.csv")
    inst_agg = aggregate_installments(inst)
    app = app.merge(inst_agg, on="sk_id_curr", how="left")
    print(f"After installments merge: {app.shape}")

    cc = _read_csv("credit_card_balance.csv")
    cc_agg = aggregate_credit_card(cc)
    app = app.merge(cc_agg, on="sk_id_curr", how="left")
    print(f"After credit_card merge: {app.shape}")

    # Fill relational-feature NaNs with 0 (means "no history in that table",
    # which is meaningfully different from "unknown" for a numeric column)
    relational_cols = [c for c in app.columns if c.startswith((
        "bureau_", "prev_", "pos_", "install_", "cc_"
    ))]
    app[relational_cols] = app[relational_cols].fillna(0)

    return app


if __name__ == "__main__":
    df = build_feature_matrix(is_train=True)
    print(f"\nFinal feature matrix shape: {df.shape}")
    print(f"Columns added by relational feature engineering: "
          f"{len([c for c in df.columns if c.startswith(('bureau_', 'prev_', 'pos_', 'install_', 'cc_'))])}")