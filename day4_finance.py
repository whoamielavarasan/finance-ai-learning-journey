import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from day3_finance import (
    load_and_clean, extract_date_features, flag_transactions
)

# ── Step 1: Encode Categoricals ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode category and day_of_week -- same logic as the synthetic
    version. category now has 19 real-world values (Groceries, Housing,
    Cash, Insurance, etc.) instead of 7 synthetic ones, so this produces
    more columns, but the mechanism is identical.
    """
    category_dummies = pd.get_dummies(df["category"], prefix="cat", dtype=int)
    dow_dummies       = pd.get_dummies(df["day_of_week"], prefix="dow", dtype=int)
    return pd.concat([df, category_dummies, dow_dummies], axis=1)


# ── Step 2: Monthly Aggregation -- FIXED for multi-year data ────────────────

def build_monthly_features(df: pd.DataFrame, drop_partial_last_month: bool = True) -> pd.DataFrame:
    """
    CRITICAL FIX vs. the synthetic Day 4 version: groups by the actual
    calendar year-month (a pandas Period, e.g. 2025-01, 2026-01) instead
    of by month number (1-12) alone. Grouping by month number alone would
    silently merge January 2025 and January 2026 into a single row --
    a real bug that only appears with real, multi-year data. Verified
    this would happen by checking df['month'].value_counts() against
    df['year_month'].value_counts() before writing this function: month=1
    showed 73 transactions total, but year_month showed 35 in 2025-01 and
    38 in 2026-01 -- two genuinely different months with different
    spending patterns, which a naive groupby('month') would have blended.

    drop_partial_last_month: the most recent calendar month in real
    transaction data is often incomplete (the statement export cuts off
    mid-month, e.g. June 2026 here has only 7 transactions vs ~40-50 for
    a full month). Including a partial month as a training row teaches
    the model an artificially low "this month's spend" figure that has
    nothing to do with actual spending behavior -- it's a data
    completeness artifact, not a real signal. Default True drops it.
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby("year_month")
        .agg(
            total_spend       = ("amount", "sum"),
            avg_transaction   = ("amount", "mean"),
            num_transactions  = ("amount", "count"),
            avoidable_spend   = ("amount", lambda x:
                                    x[df.loc[x.index, "spend_type"] == "Avoidable"].sum()),
            necessary_spend   = ("amount", lambda x:
                                    x[df.loc[x.index, "spend_type"] == "Necessary"].sum()),
            weekend_spend     = ("amount", lambda x:
                                    x[df.loc[x.index, "is_weekend"]].sum()),
        )
        .reset_index()
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    # Calendar month number (1-12) is still useful as a SEASONALITY feature
    # (e.g. "is this December" matters for predicting holiday spending),
    # so we keep it -- but ONLY as a feature alongside year_month identity,
    # never as the sole grouping key.
    monthly["month"] = monthly["year_month"].dt.month

    if drop_partial_last_month:
        last_period       = monthly["year_month"].max()
        last_month_count  = monthly.loc[monthly["year_month"] == last_period, "num_transactions"].iloc[0]
        # Compare to the median transaction count across all other months;
        # if the last month has fewer than half the typical volume, treat
        # it as partial/incomplete and drop it.
        median_count = monthly.loc[monthly["year_month"] != last_period, "num_transactions"].median()
        if last_month_count < median_count * 0.5:
            print(f"  [INFO] Dropping {last_period} as a partial month "
                  f"({last_month_count} transactions vs median {median_count:.0f})")
            monthly = monthly[monthly["year_month"] != last_period].reset_index(drop=True)

    monthly["is_outlier_month_std"] = monthly["total_spend"] > (monthly["total_spend"].mean() + 2 * monthly["total_spend"].std())
    monthly["is_outlier_month"] = (
    monthly["total_spend"] > (1.5 * monthly["total_spend"].median())
)
    return monthly


# ── Step 3: Lag and Rolling Features ─────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identical logic to the synthetic version. Now operates on TRUE
    consecutive calendar months (year_month), so a lag value genuinely
    means "last calendar month", not "whatever row happened to be above
    this one regardless of year" -- which matters once data spans a
    year boundary.
    """
    df["lag_total_spend"]      = df["total_spend"].shift(1)
    df["lag_avg_transaction"]  = df["avg_transaction"].shift(1)
    df["lag_num_transactions"] = df["num_transactions"].shift(1)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rolling_3m_spend"] = (
        df["total_spend"].rolling(window=3, min_periods=1).mean().round(2)
    )
    return df


# ── Step 4: Feature Matrix ───────────────────────────────────────────────────

def build_feature_matrix(monthly_df: pd.DataFrame):
    """
    Same mechanics as the synthetic version: predict next month's
    total_spend using this month's features plus lag/rolling history.
    """
    monthly_df = monthly_df.copy()
    monthly_df["next_month_spend"] = monthly_df["total_spend"].shift(-1)
    monthly_df = monthly_df.dropna(subset=["next_month_spend", "lag_total_spend"])

    # feature_cols = [
    #     "month", "total_spend", "avg_transaction", "num_transactions",
    #     "avoidable_spend", "necessary_spend", "weekend_spend",
    #     "lag_total_spend", "lag_avg_transaction", "lag_num_transactions",
    #     "rolling_3m_spend",
    # ]
    feature_cols = [
    "month",
    "avg_transaction",
    "num_transactions",
    "avoidable_spend",
    "necessary_spend",
    "weekend_spend",
    "lag_avg_transaction",
    "lag_num_transactions",
    "rolling_3m_spend",
]
# Removed: total_spend, lag_total_spend
# Reason: both are ~96% redundant with necessary_spend and its lag
    X = monthly_df[feature_cols]
    y = monthly_df["next_month_spend"]
    # corr = X.corr()
    # print(f"[INFO] Correlation matrix shape: {corr}")
    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target vector shape:  {y.shape}")
    print(f"\nFeature columns:\n{list(X.columns)}")
    print(f"\nFull X (real data -- only ~16 rows total, worth seeing all of it):\n{X.to_string()}")
    print(f"\nFull y:\n{y.to_string()}")
    return X, y


def split_data(X, y):
    """
    Same 80/20 chronological split, shuffle=False mandatory for time
    series -- this matters even more now since the data spans a real
    year boundary; shuffling could put a 2026 month in training and a
    2025 month in testing, which is nonsensical for forecasting.

    NOTE: with real data we only have ~16 monthly rows after dropping
    the partial last month and the first lag-less month. 80/20 split on
    16 rows means a test set of ~3 rows -- small, but this is an honest
    reflection of how little monthly history 18 months of statements
    actually gives an ML model. We proceed anyway since this is for
    learning the mechanics; Day 5 will discuss what this means for
    trusting the model's accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"\n[INFO] Training rows: {len(X_train)}")
    print(f"[INFO] Test rows:     {len(X_test)}")
    return X_train, X_test, y_train, y_test

def check_multicollinearity(X: pd.DataFrame) -> None:
    corr = X.corr().round(2)
    print("\n--- Feature Correlation Matrix ---")
    print(corr.to_string())
    print("\nHighly correlated pairs (|correlation| > 0.85):")
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) > 0.85:
                print(f"  {corr.columns[i]} vs {corr.columns[j]}: {val}")

if __name__ == "__main__":
    df = load_and_clean("transactions.csv")
    df = extract_date_features(df)
    df = flag_transactions(df)
    df = encode_categoricals(df)

    monthly = build_monthly_features(df)
    monthly = add_lag_features(monthly)
    monthly = add_rolling_features(monthly)

    print("\n--- Monthly Feature Table (real data) ---")
    print(monthly.to_string(index=False))

    X, y = build_feature_matrix(monthly)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n--- X_train ---")
    print(X_train.to_string())
    print("\n--- y_train ---")
    print(y_train.to_string())
    print("\n--- X_test ---")
    print(X_test.to_string())
    print("\n--- y_test ---")
    print(y_test.to_string())
