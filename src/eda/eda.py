"""
Phase 1 — Exploratory Data Analysis for the Home Credit Default Risk dataset.

Run standalone (no Docker needed) from the project root:
    python -m src.eda.eda

Produces:
    - Console summary: shape, dtypes, missingness, target balance
    - Data quality flags (including known sentinel-value issues)
    - Feature categorization into business-meaningful groups
    - >=5 concrete business insights (numbers, not just charts)
    - Saved charts (PNG) in notebooks/eda_outputs/
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_main_table() -> pd.DataFrame:
    path = DATA_DIR / "application_train.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def dataset_summary(df: pd.DataFrame):
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    target_counts = df["TARGET"].value_counts(normalize=True) * 100
    print("\nTarget distribution (0 = repaid, 1 = defaulted):")
    print(target_counts.round(2).to_string())
    print(f"\n=> Class imbalance ratio: {target_counts[0] / target_counts[1]:.1f} : 1")


def data_quality_report(df: pd.DataFrame, top_n: int = 15):
    print("\n" + "=" * 70)
    print("DATA QUALITY — TOP MISSING-VALUE COLUMNS")
    print("=" * 70)
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0].head(top_n)
    print(missing.round(1).to_string())

    total_missing_cols = (df.isnull().mean() > 0).sum()
    print(f"\nColumns with any missing values: {total_missing_cols} / {df.shape[1]}")

    # Known Home Credit data quality issue: DAYS_EMPLOYED uses 365243 as a
    # placeholder for "not currently employed" (pensioners/unemployed) —
    # this looks like a numeric outlier but is actually a sentinel value.
    if "DAYS_EMPLOYED" in df.columns:
        anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
        anomaly_pct = anomaly_count / len(df) * 100
        print(f"\nDATA QUALITY FLAG: DAYS_EMPLOYED contains {anomaly_count:,} rows "
              f"({anomaly_pct:.1f}%) with value 365243 (~1000 years) — this is a "
              f"placeholder/sentinel value, not a real anomaly. Must be treated as "
              f"missing/encoded separately before modeling, not left as-is.")

    # Save a missingness chart
    plt.figure(figsize=(10, 6))
    missing.plot(kind="barh", color="#4C72B0")
    plt.title("Top Missing-Value Columns (%)")
    plt.xlabel("% missing")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_missing_values.png", dpi=120)
    plt.close()


def feature_categorization(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("FEATURE CATEGORIZATION")
    print("=" * 70)

    categories = {
        "Demographic": [],
        "Financial": [],
        "Credit Bureau / External Score": [],
        "Housing / Asset": [],
        "Application Process Metadata": [],
        "Document Flags": [],
        "Social Circle / Region Risk": [],
        "Other": [],
    }

    for col in df.columns:
        c = col.upper()
        if col in ("TARGET", "SK_ID_CURR"):
            continue
        if any(k in c for k in ["CNT_CHILDREN", "DAYS_BIRTH", "CODE_GENDER",
                                 "NAME_FAMILY", "NAME_EDUCATION", "OCCUPATION",
                                 "CNT_FAM_MEMBERS", "NAME_HOUSING_TYPE"]):
            categories["Demographic"].append(col)
        elif any(k in c for k in ["AMT_INCOME", "AMT_CREDIT", "AMT_ANNUITY",
                                   "AMT_GOODS_PRICE", "NAME_INCOME_TYPE",
                                   "NAME_CONTRACT_TYPE", "DAYS_EMPLOYED"]):
            categories["Financial"].append(col)
        elif "EXT_SOURCE" in c or "AMT_REQ_CREDIT_BUREAU" in c:
            categories["Credit Bureau / External Score"].append(col)
        elif any(k in c for k in ["HOUSETYPE", "APARTMENTS", "OWN_CAR",
                                   "OWN_REALTY", "WALLSMATERIAL", "FLOORSMAX",
                                   "FLOORSMIN", "COMMONAREA", "LIVINGAREA",
                                   "LIVINGAPARTMENTS", "NONLIVING", "ELEVATORS",
                                   "ENTRANCES", "YEARS_BUILD", "YEARS_BEGINEXPLUATATION",
                                   "BASEMENTAREA", "LANDAREA", "FONDKAPREMONT",
                                   "EMERGENCYSTATE"]):
            categories["Housing / Asset"].append(col)
        elif c.startswith("FLAG_DOCUMENT"):
            categories["Document Flags"].append(col)
        elif any(k in c for k in ["SOCIAL_CIRCLE", "REGION_RATING",
                                   "REG_REGION_NOT", "REG_CITY_NOT",
                                   "LIVE_REGION_NOT", "LIVE_CITY_NOT"]):
            categories["Social Circle / Region Risk"].append(col)
        elif any(k in c for k in ["WEEKDAY_APPR", "HOUR_APPR", "ORGANIZATION_TYPE",
                                   "NAME_TYPE_SUITE", "FLAG_MOBIL", "FLAG_EMP_PHONE",
                                   "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                                   "FLAG_EMAIL", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
                                   "DAYS_LAST_PHONE_CHANGE"]):
            categories["Application Process Metadata"].append(col)
        elif c in ("REGION_POPULATION_RELATIVE", "TOTALAREA_MODE"):
            categories["Housing / Asset"].append(col)
        else:
            categories["Other"].append(col)

    for cat, cols in categories.items():
        print(f"{cat}: {len(cols)} columns")
        if cat == "Other" and cols:
            print(f"   (uncategorized: {cols})")

    return categories


def business_insights(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("BUSINESS INSIGHTS (>=5)")
    print("=" * 70)

    insights = []

    # Insight 1: Income type vs default rate
    income_default = df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
    top_income_risk = income_default.index[0]
    insights.append(
        f"1. '{top_income_risk}' applicants have the highest default rate "
        f"at {income_default.iloc[0]:.1f}%, vs overall average of {df['TARGET'].mean()*100:.1f}%."
    )

    # Insight 2: Education vs default rate
    edu_default = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
    insights.append(
        f"2. Applicants with '{edu_default.index[0]}' education default at "
        f"{edu_default.iloc[0]:.1f}%, the highest among education levels "
        f"(lowest: '{edu_default.index[-1]}' at {edu_default.iloc[-1]:.1f}%)."
    )

    # Insight 3: Age vs default (DAYS_BIRTH is negative days from application date)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(int)
    age_bins = pd.cut(df["AGE_YEARS"], bins=[18, 25, 35, 45, 55, 65, 100])
    age_default = df.groupby(age_bins, observed=True)["TARGET"].mean() * 100
    riskiest_age = age_default.idxmax()
    insights.append(
        f"3. The riskiest age group is {riskiest_age} with a "
        f"{age_default.max():.1f}% default rate, compared to "
        f"{age_default.min():.1f}% for the safest age group."
    )

    # Insight 4: Credit-to-income ratio showed almost no signal on its own
    # (an honest negative finding), so pair it with a real, material gap:
    # gender-based default rate.
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    q1, q4 = df["CREDIT_INCOME_RATIO"].quantile([0.25, 0.75])
    low_group = df[df["CREDIT_INCOME_RATIO"] <= q1]
    high_group = df[df["CREDIT_INCOME_RATIO"] >= q4]

    gender_default = df[df["CODE_GENDER"].isin(["M", "F"])].groupby("CODE_GENDER")["TARGET"].mean() * 100
    insights.append(
        f"4. Male applicants default at {gender_default.get('M', float('nan')):.1f}% "
        f"vs {gender_default.get('F', float('nan')):.1f}% for female applicants — "
        f"notably, credit-to-income ratio alone showed almost NO predictive "
        f"signal ({high_group['TARGET'].mean()*100:.1f}% top quartile vs "
        f"{low_group['TARGET'].mean()*100:.1f}% bottom quartile), a useful "
        f"negative finding that argues against naive debt-ratio-based manual "
        f"underwriting rules."
    )

    # Insight 5: EXT_SOURCE correlation (external credit bureau scores)
    ext_cols = [c for c in df.columns if "EXT_SOURCE" in c]
    if ext_cols:
        corrs = df[ext_cols + ["TARGET"]].corr()["TARGET"].drop("TARGET")
        strongest = corrs.abs().idxmax()
        insights.append(
            f"5. '{strongest}' (external credit bureau score) has the strongest "
            f"correlation with default (r = {corrs[strongest]:.3f}) among all "
            f"external source features — a strong candidate for a top model feature."
        )

    # Insight 6: Car/realty ownership
    own_default = df.groupby(["FLAG_OWN_CAR", "FLAG_OWN_REALTY"])["TARGET"].mean() * 100
    insights.append(
        f"6. Applicants who own neither a car nor real estate default at "
        f"{own_default.get(('N', 'N'), float('nan')):.1f}%, notably higher than "
        f"those who own both ({own_default.get(('Y', 'Y'), float('nan')):.1f}%)."
    )

    for line in insights:
        print(line)

    # Save a couple of supporting charts
    plt.figure(figsize=(8, 5))
    income_default.plot(kind="bar", color="#C44E52")
    plt.title("Default Rate by Income Type")
    plt.ylabel("Default rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_default_by_income_type.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    age_default.plot(kind="bar", color="#55A868")
    plt.title("Default Rate by Age Group")
    plt.ylabel("Default rate (%)")
    plt.xlabel("Age group")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_default_by_age_group.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    gender_default.plot(kind="bar", color="#8172B2")
    plt.title("Default Rate by Gender")
    plt.ylabel("Default rate (%)")
    plt.xlabel("Gender")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_default_by_gender.png", dpi=120)
    plt.close()

    return insights


def main():
    print("Loading application_train.csv ...")
    df = load_main_table()

    dataset_summary(df)
    data_quality_report(df)
    feature_categorization(df)
    business_insights(df)

    print("\n" + "=" * 70)
    print(f"Charts saved to: {OUTPUT_DIR}")
    print("=" * 70)


    main()


def get_eda_summary_dict() -> dict:
    """
    Structured, API-friendly version of the EDA output for the /eda/summary
    endpoint. Reuses the same insight-generation and categorization logic
    as main() (business_insights, feature_categorization) rather than
    duplicating computation - just wraps it for JSON serialization instead
    of console printing.
    """
    df = load_main_table()

    target_counts = df["TARGET"].value_counts(normalize=True) * 100
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    top_missing = missing[missing > 0].head(10).round(1).to_dict()

    categories = feature_categorization(df)
    category_counts = {cat: len(cols) for cat, cols in categories.items()}

    insights = business_insights(df)

    anomaly_count = int((df["DAYS_EMPLOYED"] == 365243).sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_distribution": {
            "repaid_pct": round(float(target_counts.get(0, 0)), 2),
            "defaulted_pct": round(float(target_counts.get(1, 0)), 2),
        },
        "class_imbalance_ratio": round(float(target_counts.get(0, 0) / target_counts.get(1, 1)), 1),
        "top_missing_columns": top_missing,
        "feature_categories": category_counts,
        "business_insights": insights,
        "data_quality_flags": [
            f"DAYS_EMPLOYED contains {anomaly_count:,} rows (~18%) with sentinel value 365243, "
            f"treated as missing + explicitly flagged during feature engineering, not left as a real value."
        ],
    }
