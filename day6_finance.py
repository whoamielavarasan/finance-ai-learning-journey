from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import os
import anthropic

from day3_finance import (
    load_and_clean, extract_date_features, flag_transactions,
    get_category_summary, get_inefficiency_score,
    get_avoidable_transactions
)
from day4_finance import (
    encode_categoricals, build_monthly_features,
    add_lag_features, add_rolling_features, build_feature_matrix,
    split_data
)
from day5_finance import train_random_forest, evaluate_model


FEATURE_COLS = [
    "month", "avg_transaction", "num_transactions",
    "avoidable_spend", "necessary_spend", "weekend_spend",
    "lag_avg_transaction", "lag_num_transactions", "rolling_3m_spend"
]


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["cat_mean"] = df.groupby("category")["amount"].transform("mean")
    df["cat_std"]  = df.groupby("category")["amount"].transform("std").fillna(1)
    df["z_score"]  = ((df["amount"] - df["cat_mean"]) / df["cat_std"]).round(2)
    return (
        df[df["z_score"] > z_threshold]
        [["date", "description", "amount", "category", "spend_type", "z_score"]]
        .sort_values("z_score", ascending=False)
        .reset_index(drop=True)
    )


def detect_monthly_trends(monthly: pd.DataFrame) -> dict:
    max_month  = monthly.loc[monthly["total_spend"].idxmax()]
    min_month  = monthly.loc[monthly["total_spend"].idxmin()]
    last_3     = monthly.tail(3)["total_spend"].values
    if last_3[-1] > last_3[0] * 1.10:
        trend = "INCREASING"
    elif last_3[-1] < last_3[0] * 0.90:
        trend = "DECREASING"
    else:
        trend = "STABLE"
    recent     = monthly.iloc[-1]["total_spend"]
    prior      = monthly.iloc[-2]["total_spend"]
    mom_change = round((recent - prior) / prior * 100, 1)
    return {
        "highest_month":        str(max_month["year_month"]),
        "highest_spend":        round(max_month["total_spend"], 2),
        "lowest_month":         str(min_month["year_month"]),
        "lowest_spend":         round(min_month["total_spend"], 2),
        "recent_trend":         trend,
        "mom_change_pct":       mom_change,
        "avg_monthly_spend":    round(monthly["total_spend"].mean(), 2),
        "median_monthly_spend": round(monthly["total_spend"].median(), 2),
    }


def predict_next_month(rf_model, monthly: pd.DataFrame, feature_cols: list) -> float:
    latest_row = monthly[feature_cols].iloc[[-1]]
    latest_row1 = monthly[feature_cols].iloc[-1]
    print("\nEla",latest_row,"\nTejashri", latest_row1,"\n")
    return round(rf_model.predict(latest_row)[0], 2)


def generate_financial_insights(
    category_summary: pd.DataFrame,
    inefficiency_score: dict,
    predicted_next_month: float,
    rf_mae: float
) -> str:
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    category_text = category_summary[
        ["category", "spend_type", "total_spend", "pct_of_total"]
    ].to_string(index=False)

    prompt = f"""
You are a personal finance advisor analyzing real spending data.
Here is a summary of spending over the past 18 months:

CATEGORY BREAKDOWN:
{category_text}

EFFICIENCY METRICS:
- Total spend: ${inefficiency_score['total_spend']:,.2f}
- Necessary spend: ${inefficiency_score['necessary_spend']:,.2f}
- Avoidable spend: ${inefficiency_score['avoidable_spend']:,.2f}
- Inefficiency score: {inefficiency_score['inefficiency_pct']}%
- Verdict: {inefficiency_score['verdict']}

ML PREDICTION:
- Predicted next month spend: ${predicted_next_month:,.2f}
- Model margin of error (MAE): ${rf_mae:,.2f}
- Prediction range: ${predicted_next_month - rf_mae:,.2f} to \
${predicted_next_month + rf_mae:,.2f}

Please provide:
1. Three specific, actionable insights based on this person's
   ACTUAL spending patterns (not generic advice)
2. The single biggest financial risk you see in this data
3. One concrete recommendation to reduce the inefficiency score
4. A one-sentence honest assessment of the ML prediction's reliability

Keep the response concise, direct, and grounded in the numbers above.
Do not give generic budgeting advice -- only insights specific to
what these numbers reveal.
"""
    response = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 1000,
        messages   = [{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def print_full_report(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    rf_model,
    rf_mae: float
) -> None:
    sep = "=" * 65

    print(sep)
    print("       PERSONAL FINANCE — COMPLETE INTELLIGENCE REPORT")
    print(sep)

    # ── Category Summary ─────────────────────────────────────────────
    print("\n[ 1. CATEGORY BREAKDOWN — 18 MONTHS ]")
    cat_summary = get_category_summary(df)
    print(cat_summary.to_string(index=False))

    # ── Efficiency Score ──────────────────────────────────────────────
    print("\n[ 2. EFFICIENCY SCORE ]")
    score = get_inefficiency_score(df)
    for key, value in score.items():
        print(f"  {key:<22}: {value}")

    # ── Monthly Trends ────────────────────────────────────────────────
    print("\n[ 3. MONTHLY TRENDS ]")
    trends = detect_monthly_trends(monthly)
    for key, value in trends.items():
        print(f"  {key:<25}: {value}")

    # ── Anomaly Detection ─────────────────────────────────────────────
    print("\n[ 4. ANOMALOUS TRANSACTIONS (z-score > 2.0) ]")
    anomalies = detect_anomalies(df)
    if anomalies.empty:
        print("  No anomalous transactions detected.")
    else:
        print(f"  {len(anomalies)} unusual transactions found:")
        print(anomalies.to_string(index=False))

    # ── Avoidable Spend ───────────────────────────────────────────────
    print("\n[ 5. TOP AVOIDABLE TRANSACTIONS ]")
    avoidable = get_avoidable_transactions(df).head(10)
    print(avoidable.to_string(index=False))

    # ── ML Prediction ─────────────────────────────────────────────────
    print("\n[ 6. NEXT MONTH PREDICTION ]")
    prediction = predict_next_month(rf_model, monthly, FEATURE_COLS)
    print(f"  Predicted spend:    ${prediction:,.2f}")
    print(f"  Margin of error:  ±${rf_mae:,.2f}")
    print(f"  Expected range:     ${prediction - rf_mae:,.2f} — ${prediction + rf_mae:,.2f}")

    # ── AI Insights ───────────────────────────────────────────────────
    print("\n[ 7. AI-GENERATED INSIGHTS ]")
    print("  Calling Claude API...")
    # try:
    #     insights = generate_financial_insights(cat_summary, score, prediction, rf_mae)
    #     print(insights)
    # except Exception as e:
    #     print(f"  [WARN] API call failed: {e}")
    #     print("  Check ANTHROPIC_API_KEY environment variable.")
    
    print("\n" + sep)


if __name__ == "__main__":
    # ── Pipeline ──────────────────────────────────────────────────────
    df = load_and_clean("transactions.csv")
    df = extract_date_features(df)
    df = flag_transactions(df)
    df = encode_categoricals(df)

    monthly = build_monthly_features(df)
    monthly = add_lag_features(monthly)
    monthly = add_rolling_features(monthly)

    X, y = build_feature_matrix(monthly)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Train and evaluate RF to get the MAE we'll report ────────────
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_mae = rf_results["mae"]

    # ── Full report ───────────────────────────────────────────────────
    print_full_report(df, monthly, rf_model, rf_mae)