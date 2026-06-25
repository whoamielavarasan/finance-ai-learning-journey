import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from day3_finance import load_and_clean, extract_date_features, flag_transactions
from day4_finance import (
    check_multicollinearity, encode_categoricals, build_monthly_features,
    add_lag_features, add_rolling_features,
    build_feature_matrix, split_data
)


# ── Model Training ────────────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train) -> LinearRegression:
    """
    Fits a straight-line (well, straight-hyperplane, in 11 dimensions)
    formula to the training data. Simple, fast, and -- importantly --
    fully interpretable: every weight tells you exactly how much that
    feature pushes the prediction up or down.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    A completely different approach: instead of one formula, it builds
    many decision trees (n_estimators=100 means 100 trees), each trained
    on a slightly different random subset of the data, then AVERAGES
    their predictions.

    Decision tree intuition: imagine a flowchart of yes/no questions
    like "is total_spend > $8,000? -> if yes, is lag_total_spend >
    $9,000? -> ..." eventually landing on a predicted number. A single
    tree overfits easily (memorizes quirks instead of learning real
    patterns). A "forest" of many slightly-different trees, averaged
    together, is more stable -- this is called an ENSEMBLE method.

    random_state=42 fixes the randomness so your results are
    reproducible every time you run this script (same "seed" concept
    from Day 4's synthetic data generator).
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# ── Model Evaluation ──────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    predictions = model.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Predictions vs Actual:")
    comparison = pd.DataFrame({
        "actual":     y_test.values,
        "predicted":  predictions.round(2),
        "error":      (y_test.values - predictions).round(2),
    })
    print(comparison.to_string(index=False))
    print(f"\nMAE:  ${mae:,.2f}   (average dollar miss)")
    print(f"RMSE: ${rmse:,.2f}   (dollar miss, big errors punished harder)")

    return {"model": model_name, "mae": round(mae, 2), "rmse": round(rmse, 2)}


def show_feature_importance(model, feature_names, model_name: str):
    """
    Linear Regression: each weight (coefficient) tells you the direct
    relationship -- "for every $1 increase in this feature, the
    prediction changes by this many dollars" (holding everything else
    constant).

    Random Forest: importances are NOT weights in a formula -- they
    represent "how often, and how usefully, did this feature get used
    to split a decision tree" across all 100 trees. Higher = the model
    relied on it more.
    """
    print(f"\n--- {model_name} Feature Importance ---")
    if hasattr(model, "coef_"):
        importance = pd.DataFrame({
            "feature":     feature_names,
            "coefficient": model.coef_.round(2)
        }).sort_values("coefficient", key=abs, ascending=False)
        print(importance.to_string(index=False))
    elif hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_.round(4)
        }).sort_values("importance", ascending=False)
        print(importance.to_string(index=False))


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean("transactions.csv")
    df = extract_date_features(df)
    df = flag_transactions(df)
    df = encode_categoricals(df)

    monthly = build_monthly_features(df)
    monthly = add_lag_features(monthly)
    monthly = add_rolling_features(monthly)

    X, y = build_feature_matrix(monthly)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n--- check_multicollinearity ---")
    check_multicollinearity(X)    # ← add this line
    
    # ── Train both models ────────────────────────────────────────────────────
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # ── Evaluate both ─────────────────────────────────────────────────────────
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # ── Feature importance for both ──────────────────────────────────────────
    show_feature_importance(lr_model, X.columns, "Linear Regression")
    show_feature_importance(rf_model, X.columns, "Random Forest")

    # ── Side-by-side comparison ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   MODEL COMPARISON")
    print("=" * 50)
    comparison = pd.DataFrame([lr_results, rf_results])
    print(comparison.to_string(index=False))

    better = comparison.loc[comparison["mae"].idxmin(), "model"]
    print(f"\nLower MAE = better on this test set: {better}")
    print("(With only 3 test rows, treat this comparison as informative,")
    print(" not definitive -- more on this in the evaluation discussion.)")
    