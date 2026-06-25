"""
Personal Finance AI — Main Application Entry Point
Day 7: Full CLI integration of the 6-day pipeline

Usage:
  py finance_app.py --report
  py finance_app.py --predict
  py finance_app.py --anomalies --threshold 3.0
  py finance_app.py --retrain --report
"""

import argparse
import joblib
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Import from all previous days ────────────────────────────────────────────
from day3_finance import (
    load_and_clean, extract_date_features, flag_transactions,
    get_category_summary, get_inefficiency_score,
    get_avoidable_transactions
)
from day4_finance import (
    encode_categoricals, build_monthly_features,
    add_lag_features, add_rolling_features,
    build_feature_matrix, split_data
)
from day5_finance import train_random_forest, evaluate_model
from day6_finance import (
    detect_anomalies, detect_monthly_trends,
    predict_next_month, generate_financial_insights,
    print_full_report, FEATURE_COLS
)

MODEL_PATH = "finance_model.joblib"
DATA_PATH  = "transactions.csv"


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model, filepath: str) -> None:
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved: {filepath}")


def load_model_from_disk(filepath: str):
    model = joblib.load(filepath)
    print(f"[INFO] Model loaded: {filepath}")
    return model


def model_exists(filepath: str) -> bool:
    return os.path.exists(filepath)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str, model_path: str, force_retrain: bool = False) -> dict:
    """Runs full pipeline, returns all artifacts needed by CLI commands."""

    df = load_and_clean(data_path)
    df = extract_date_features(df)
    df = flag_transactions(df)
    df = encode_categoricals(df)

    monthly = build_monthly_features(df)
    monthly = add_lag_features(monthly)
    monthly = add_rolling_features(monthly)

    X, y = build_feature_matrix(monthly)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if model_exists(model_path) and not force_retrain:
        rf_model  = load_model_from_disk(model_path)
        rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest (loaded)")
    else:
        print("[INFO] Training Random Forest model...")
        rf_model  = train_random_forest(X_train, y_train)
        rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        save_model(rf_model, model_path)

    return {
        "df":       df,
        "monthly":  monthly,
        "rf_model": rf_model,
        "rf_mae":   rf_results["mae"],
    }


# ── CLI Commands ──────────────────────────────────────────────────────────────

def cmd_report(artifacts: dict) -> None:
    """--report: Full financial intelligence report."""
    print_full_report(
        artifacts["df"],
        artifacts["monthly"],
        artifacts["rf_model"],
        artifacts["rf_mae"]
    )


def cmd_predict(artifacts: dict) -> None:
    """--predict: Next month spend prediction only."""
    prediction = predict_next_month(
        artifacts["rf_model"],
        artifacts["monthly"],
        FEATURE_COLS
    )
    mae = artifacts["rf_mae"]
    print("\n" + "=" * 50)
    print("   NEXT MONTH SPEND PREDICTION")
    print("=" * 50)
    print(f"  Predicted spend:   ${prediction:,.2f}")
    print(f"  Margin of error: ±${mae:,.2f}")
    print(f"  Expected range:    ${prediction - mae:,.2f} — ${prediction + mae:,.2f}")
    print("=" * 50)


def cmd_anomalies(artifacts: dict, threshold: float) -> None:
    """--anomalies: Show statistically unusual transactions."""
    anomalies = detect_anomalies(artifacts["df"], z_threshold=threshold)
    print(f"\n{'=' * 55}")
    print(f"   ANOMALOUS TRANSACTIONS (z-score > {threshold})")
    print(f"{'=' * 55}")
    if anomalies.empty:
        print("  No anomalous transactions detected at this threshold.")
    else:
        print(f"  {len(anomalies)} unusual transactions found:\n")
        print(anomalies.to_string(index=False))
    print("=" * 55)


# ── Argument Parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Personal Finance AI — Analysis & Prediction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py finance_app.py --report
  py finance_app.py --predict
  py finance_app.py --anomalies
  py finance_app.py --anomalies --threshold 3.0
  py finance_app.py --retrain
  py finance_app.py --retrain --report
  py finance_app.py --data my_transactions.csv --report
        """
    )
    parser.add_argument("--report",     action="store_true",
                        help="Full financial intelligence report")
    parser.add_argument("--predict",    action="store_true",
                        help="Next month spend prediction only")
    parser.add_argument("--anomalies",  action="store_true",
                        help="Show anomalous transactions")
    parser.add_argument("--retrain",    action="store_true",
                        help="Force model retraining from scratch")
    parser.add_argument("--threshold",  type=float, default=2.0,
                        help="Z-score threshold for anomaly detection (default: 2.0)")
    parser.add_argument("--data",       type=str, default=DATA_PATH,
                        help=f"Transactions CSV path (default: {DATA_PATH})")
    parser.add_argument("--model",      type=str, default=MODEL_PATH,
                        help=f"Model file path (default: {MODEL_PATH})")
    return parser


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    # If no command specified, show help and exit
    if not any([args.report, args.predict, args.anomalies, args.retrain]):
        parser.print_help()
        print("\n[HINT] Try: py finance_app.py --report")
        exit(0)

    # Run the pipeline once -- all commands share the same artifacts
    artifacts = run_pipeline(
        data_path     = args.data,
        model_path    = args.model,
        force_retrain = args.retrain,
    )

    # Execute whichever commands were requested
    # Multiple flags can be combined: --predict --anomalies runs both
    if args.predict:
        cmd_predict(artifacts)

    if args.anomalies:
        cmd_anomalies(artifacts, threshold=args.threshold)

    if args.report:
        cmd_report(artifacts)