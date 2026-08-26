"""
ML Forecaster — Random Forest return predictions with walk-forward validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    ALL_TICKERS,
    ML_TRAIN_END_DATE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_RANDOM_STATE,
)


# ── Feature Engineering ──────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "Lag_1M",
    "Momentum_3M",
    "Momentum_6M",
    "Volatility_3M",
    "Volatility_6M",
    "Market_Return_1M",
]


def build_ml_dataset(
    monthly_returns: pd.DataFrame,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a panel dataset with lag/momentum/volatility features
    and a 1-month forward return target.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    market_returns = monthly_returns.mean(axis=1)
    frames = []

    for ticker in tickers:
        ret = monthly_returns[ticker]
        feat = pd.DataFrame(index=ret.index)
        feat["Ticker"] = ticker
        feat["Lag_1M"] = ret.shift(1)
        feat["Momentum_3M"] = ret.rolling(3).mean().shift(1)
        feat["Momentum_6M"] = ret.rolling(6).mean().shift(1)
        feat["Volatility_3M"] = ret.rolling(3).std().shift(1)
        feat["Volatility_6M"] = ret.rolling(6).std().shift(1)
        feat["Market_Return_1M"] = market_returns.shift(1)
        feat["Target_Return_1M"] = ret
        frames.append(feat)

    return pd.concat(frames).dropna()


# ── Model Training & Prediction ──────────────────────────────────────────────

def train_and_predict(
    ml_dataset: pd.DataFrame,
    train_end: str = ML_TRAIN_END_DATE,
) -> dict:
    """
    Train a Random Forest on data up to `train_end` and generate
    out-of-sample predictions for the remaining period.

    Returns
    -------
    dict with keys:
        "model"       : fitted RandomForestRegressor
        "predictions" : pd.DataFrame with Ticker, Target, Predicted
        "metrics"     : dict of evaluation metrics
        "feature_importance" : pd.Series
    """
    train_mask = ml_dataset.index <= train_end
    test_mask = ml_dataset.index > train_end

    X_train = ml_dataset.loc[train_mask, FEATURE_COLUMNS]
    y_train = ml_dataset.loc[train_mask, "Target_Return_1M"]
    X_test = ml_dataset.loc[test_mask, FEATURE_COLUMNS]
    y_test = ml_dataset.loc[test_mask, "Target_Return_1M"]

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    dir_acc = float((np.sign(y_pred) == np.sign(y_test)).mean())

    predictions = ml_dataset.loc[test_mask, ["Ticker", "Target_Return_1M"]].copy()
    predictions["Predicted_Return_1M"] = y_pred

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    return {
        "model": model,
        "predictions": predictions,
        "metrics": {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Directional_Accuracy": round(dir_acc, 4),
        },
        "feature_importance": importance,
    }


def format_forecast_summary(result: dict) -> str:
    """Human-readable summary of ML forecast results."""
    m = result["metrics"]
    lines = [
        "**ML Forecast Results (Random Forest):**\n",
        f"- MAE: {m['MAE']}",
        f"- RMSE: {m['RMSE']}",
        f"- R²: {m['R2']}",
        f"- Directional Accuracy: {m['Directional_Accuracy']:.2%}",
        "\n**Feature Importance:**",
    ]
    for feat, imp in result["feature_importance"].items():
        lines.append(f"- {feat}: {imp:.3f}")

    # Per-ticker predictions (latest month)
    preds = result["predictions"]
    latest_date = preds.index.max()
    latest = preds.loc[preds.index == latest_date].sort_values(
        "Predicted_Return_1M", ascending=False
    )
    lines.append(f"\n**Latest Predictions ({latest_date.strftime('%Y-%m')}):**")
    for _, row in latest.iterrows():
        actual = row["Target_Return_1M"] * 100
        predicted = row["Predicted_Return_1M"] * 100
        lines.append(
            f"- {row['Ticker']}: predicted {predicted:+.2f}% "
            f"(actual {actual:+.2f}%)"
        )

    return "\n".join(lines)
