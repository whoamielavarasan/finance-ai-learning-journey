import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

CATEGORY_FLAGS = {
    "Food":           "Necessary",
    "Housing":        "Necessary",
    "Transportation": "Necessary",
    "Healthcare":     "Necessary",
    "Utilities":      "Necessary",
    "Entertainment":  "Avoidable",
    "Shopping":       "Avoidable",
    "Dining Out":     "Avoidable",
    "Subscriptions":  "Avoidable",
    "Travel":         "Avoidable",
}

# ── Ingestion & Cleaning (carried forward from Day 2) ────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw rows loaded: {len(df)}")
    df = parse_dates_amount(df)
    df = normalize_text(df)
    df = drop_invalid_rows(df)
    df = drop_exact_duplicates(df)
    df = reset_index(df)
    print(f"[INFO] Clean rows after pipeline: {len(df)}")
    return df

def parse_dates_amount(df: pd.DataFrame) -> pd.DataFrame:
    df["date"]   = pd.to_datetime(df["date"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df["description"] = df["description"].str.strip()
    df["category"]    = df["category"].str.strip().str.title()
    return df

def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    before  = len(df)
    cleaned = df.dropna(subset=["date", "amount", "category"])
    dropped = before - len(cleaned)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} invalid row(s).")
    return cleaned

def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before  = len(df)
    cleaned = df.drop_duplicates(keep="first")
    dropped = before - len(cleaned)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} exact duplicate row(s).")
    return cleaned

def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.month_name()
    df["day"]         = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"]    = df["date"].dt.day_name()
    df["is_weekend"]  = df["day_of_week"].isin([5, 6])
    return df

# ── Analytics Engine ─────────────────────────────────────────────────────────

def flag_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns spend_type ONLY when it is missing or blank.
    transactions.csv already has a correct, fully-populated spend_type
    column (derived by categorize_merchants.py for credit card rows and
    by parse_salary_real.py for salary account rows). This function must
    not overwrite that. It exists only to backfill spend_type for any
    NEW rows added later that don't have one yet (e.g. if you paste in
    a fresh batch of transactions with only a 'category' column filled in).
    """
    if "spend_type" not in df.columns:
        df["spend_type"] = pd.NA

    missing_mask = df["spend_type"].isna() | (df["spend_type"].astype(str).str.strip() == "")
    df.loc[missing_mask, "spend_type"] = df.loc[missing_mask, "category"].map(CATEGORY_FLAGS)
    df["spend_type"] = df["spend_type"].fillna("Uncategorized")
    return df

def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_all = df["amount"].sum()
    return (
        df.groupby(["category", "spend_type"])
        .agg(
            total_spend      = ("amount", "sum"),
            avg_spend        = ("amount", "mean"),
            num_transactions = ("amount", "count")
        )
        .reset_index()
        .assign(pct_of_total=lambda d: (d["total_spend"] / total_all * 100).round(2))
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )

def get_top_transactions(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return (
        df[["date", "description", "amount", "category", "spend_type"]]
        .sort_values("amount", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

def get_spend_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_all = df["amount"].sum()
    return (
        df.groupby("spend_type")
        .agg(
            total_spend      = ("amount", "sum"),
            num_transactions = ("amount", "count")
        )
        .reset_index()
        .assign(pct_of_total=lambda d: (d["total_spend"] / total_all * 100).round(2))
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )

def get_avoidable_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["spend_type"] == "Avoidable"]
        [["date", "description", "amount", "category"]]
        .sort_values("amount", ascending=False)
        .reset_index(drop=True)
    )

def get_inefficiency_score(df: pd.DataFrame) -> dict:
    total_spend = df["amount"].sum()
    if total_spend <= 0:
        raise ValueError(
            f"Cannot compute inefficiency score: total spend is {total_spend}. "
            f"Expected a positive value."
        )
    avoidable_spend = df[df["spend_type"] == "Avoidable"]["amount"].sum()
    necessary_spend = df[df["spend_type"] == "Necessary"]["amount"].sum()
    score           = round(avoidable_spend / total_spend * 100, 2)
    return {
        "total_spend":      round(total_spend, 2),
        "necessary_spend":  round(necessary_spend, 2),
        "avoidable_spend":  round(avoidable_spend, 2),
        "inefficiency_pct": score,
        "verdict":          "HIGH" if score > 40 else "MODERATE" if score > 20 else "LOW"
    }

def get_weekend_spend_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_all = df["amount"].sum()
    summary = (
        df.groupby(["is_weekend", "spend_type"])
        .agg(
            total_spend      = ("amount", "sum"),
            num_transactions = ("amount", "count")
        )
        .reset_index()
        .assign(pct_of_total=lambda d:(d["total_spend"] / total_all * 100).round(2))
        .sort_values("is_weekend", ascending=False)
        .reset_index(drop=True)
        )
    return summary

def get_savings_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["savings_opportunity"] = np.where(
        result["spend_type"] == "Avoidable",
        result["amount"],
        0.0
    )
    return result.sort_values("savings_opportunity", ascending=False)

def print_report(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("   PERSONAL FINANCE — MONTHLY ANALYTICS REPORT")
    print("=" * 60)
    # print("\n[ CATEGORY BREAKDOWN ]")
    # print(get_category_summary(df).to_string(index=False))
    # print("\n[ NECESSARY vs AVOIDABLE ]")
    # print(get_spend_type_summary(df).to_string(index=False))
    # print("\n[ TOP 5 TRANSACTIONS ]")
    # print(get_top_transactions(df, n=5).to_string(index=False))
    # print("\n[ AVOIDABLE TRANSACTIONS ]")
    # print(get_avoidable_transactions(df).to_string(index=False))
    # print("\n[ INEFFICIENCY SCORE ]")
    # score = get_inefficiency_score(df)
    # for key, value in score.items():
    #     print(f"  {key:<20}: {value}")
    # print("\n" + "=" * 60)
    print("\n[ WEEKEND SPEND SUMMARY ]")
    print(get_weekend_spend_summary(df).to_string(index=False))
    # print("\n[ SAVINGS OPPORTUNITIES ]")
    # print(get_savings_opportunities(df)[["date", "description", "amount", "category", "savings_opportunity","spend_type"]].to_string(index=False))

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean("transactions.csv")
    df = extract_date_features(df)
    df = flag_transactions(df)
    # print_report(df)