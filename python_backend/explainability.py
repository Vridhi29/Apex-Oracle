"""
Apex-Oracle — Explainability Layer
Generates natural-language prediction narratives using SHAP values
and feature importance from multiple models.
"""
import json
import numpy as np
import pandas as pd
import joblib
from loguru import logger

from config import DATA_DIR, MODELS_DIR, STOCK_UNIVERSE


# Human-readable feature name mapping
FEATURE_DISPLAY_NAMES = {
    "rsi": "RSI momentum",
    "macd": "MACD trend signal",
    "macd_diff": "MACD histogram",
    "bb_width": "Bollinger Band width",
    "atr": "average true range",
    "ema_12": "12-day EMA",
    "ema_26": "26-day EMA",
    "sma_20": "20-day moving average",
    "sma_50": "50-day moving average",
    "volume_ratio": "trading volume ratio",
    "momentum": "price momentum",
    "roc": "rate of change",
    "daily_return": "daily returns",
    "volatility_20d": "20-day volatility",
    "sentiment_score": "news sentiment",
    "sentiment_positive": "positive sentiment",
    "sentiment_negative": "negative sentiment",
    "close_lag_1": "yesterday's price",
    "close_lag_2": "price 2 days ago",
    "return_lag_1": "yesterday's return",
    "volume_lag_1": "yesterday's volume",
}


def get_shap_explanations(ticker: str) -> dict:
    """
    Compute SHAP values for the XGBoost model predictions.
    Returns top contributing features with their impact.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed")
        return {}

    safe_name = ticker.replace(".", "_")
    model_path = MODELS_DIR / f"xgboost_{safe_name}.pkl"

    if not model_path.exists():
        return {}

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    features_path = DATA_DIR / safe_name / "features.csv"
    if not features_path.exists():
        return {}

    df = pd.read_csv(features_path)
    X = df[feature_cols].values

    # Use the last 100 rows as background
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[-1:])

    # Get top features
    feature_contributions = {}
    for i, col in enumerate(feature_cols):
        feature_contributions[col] = float(shap_values[0][i])

    # Sort by absolute impact
    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "top_features": [
            {
                "feature": f,
                "display_name": FEATURE_DISPLAY_NAMES.get(f, f.replace("_", " ")),
                "impact": round(v, 4),
                "direction": "positive" if v > 0 else "negative",
            }
            for f, v in sorted_features[:8]
        ],
        "model": "XGBoost SHAP",
    }


def get_rf_importance(ticker: str) -> dict:
    """Get Random Forest feature importance."""
    from random_forest_model import RandomForestModel
    rf = RandomForestModel()
    return rf.get_feature_importance(ticker)


def get_lr_coefficients(ticker: str) -> dict:
    """Get Logistic Regression coefficients."""
    from logistic_regression_model import LogisticRegressionModel
    lr = LogisticRegressionModel()
    return lr.get_coefficients(ticker)


def generate_narrative(ticker: str, prediction_result: dict) -> str:
    """
    Generate a human-readable prediction narrative.
    This is the key differentiator — turns numbers into stories.
    """
    pred = prediction_result.get("prediction", {})
    regime = prediction_result.get("regime", {})
    agreement = prediction_result.get("agreement", {})

    price = pred.get("price", 0)
    current = pred.get("current_price", 0)
    change = pred.get("change_pct", 0)
    direction = pred.get("direction", "UNKNOWN")
    confidence = pred.get("confidence", 0)
    conf_label = pred.get("confidence_label", "Unknown")
    current_regime = regime.get("current", "UNKNOWN")

    company = STOCK_UNIVERSE.get(ticker, ticker)
    models_agree = agreement.get("models_agreeing", 0)
    total_models = agreement.get("total_models", 8)
    class_vote = agreement.get("classification_vote", "N/A")
    conflict = agreement.get("conflict_flag", False)

    # Get SHAP explanations
    shap_data = get_shap_explanations(ticker)
    top_features = shap_data.get("top_features", [])

    # Build narrative
    direction_word = "rise" if direction == "UP" else "fall"
    arrow = "📈" if direction == "UP" else "📉"

    narrative = f"{arrow} **{company}** is predicted to close at **₹{price:,.2f}** tomorrow "
    narrative += f"({'+' if change > 0 else ''}{change:.1f}%). "
    narrative += f"Confidence: **{confidence:.0%}** ({conf_label}).\n\n"

    narrative += f"🏛️ Market regime: **{current_regime}**. "
    narrative += f"Direction consensus: **{direction}** ({models_agree}/{total_models} models agree). "
    narrative += f"Classifier vote: {class_vote}.\n\n"

    if conflict:
        narrative += "⚠️ **CONFLICT ALERT**: Regression models and classifiers disagree on direction. "
        narrative += "Confidence has been reduced.\n\n"

    # Add factor explanations
    if top_features:
        narrative += "📊 **Key driving factors**:\n"
        for i, feat in enumerate(top_features[:5], 1):
            impact_str = f"+{feat['impact']:.3f}" if feat["impact"] > 0 else f"{feat['impact']:.3f}"
            emoji = "🟢" if feat["direction"] == "positive" else "🔴"
            narrative += f"  {i}. {emoji} {feat['display_name'].capitalize()} ({impact_str})\n"

    return narrative


def generate_full_explanation(ticker: str, prediction_result: dict) -> dict:
    """
    Generate complete explanation package: narrative + SHAP + RF importance + LR coefficients.
    """
    narrative = generate_narrative(ticker, prediction_result)
    shap_data = get_shap_explanations(ticker)
    rf_importance = get_rf_importance(ticker)
    lr_coefficients = get_lr_coefficients(ticker)

    explanation = {
        "narrative": narrative,
        "shap": shap_data,
        "rf_importance": rf_importance,
        "lr_coefficients": lr_coefficients,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    # Save explanation
    safe_name = ticker.replace(".", "_")
    path = DATA_DIR / safe_name / "latest_explanation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2, ensure_ascii=False)

    return explanation
