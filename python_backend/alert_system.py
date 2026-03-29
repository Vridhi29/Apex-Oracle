"""
Apex-Oracle — Alert System
Generates alerts based on prediction confidence, regime transitions, and model conflicts.
"""
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

from config import DATA_DIR, HIGH_CONFIDENCE_THRESHOLD, ALERT_MIN_MODELS_AGREE


def generate_alerts(ticker: str, prediction_result: dict) -> list:
    """Generate alerts based on prediction analysis."""
    alerts = []
    pred = prediction_result.get("prediction", {})
    regime = prediction_result.get("regime", {})
    agreement = prediction_result.get("agreement", {})
    company = prediction_result.get("company", ticker)

    confidence = pred.get("confidence", 0)
    direction = pred.get("direction", "UNKNOWN")
    models_agreeing = agreement.get("models_agreeing", 0)
    total_models = agreement.get("total_models", 8)
    conflict = agreement.get("conflict_flag", False)
    current_regime = regime.get("current", "UNKNOWN")
    change_pct = pred.get("change_pct", 0)

    now = datetime.now().isoformat()

    # 1. High confidence + strong agreement alert
    if confidence >= HIGH_CONFIDENCE_THRESHOLD and models_agreeing >= ALERT_MIN_MODELS_AGREE:
        alerts.append({
            "type": "HIGH_CONFIDENCE",
            "severity": "success",
            "ticker": ticker,
            "company": company,
            "title": f"🎯 Strong {direction} signal for {company}",
            "message": (
                f"{models_agreeing}/{total_models} models agree on {direction} "
                f"with {confidence:.0%} confidence. "
                f"Expected change: {'+' if change_pct > 0 else ''}{change_pct:.1f}%"
            ),
            "timestamp": now,
        })

    # 2. Conflict alert (regression vs classification disagree)
    if conflict:
        alerts.append({
            "type": "CONFLICT",
            "severity": "warning",
            "ticker": ticker,
            "company": company,
            "title": f"⚠️ Model conflict for {company}",
            "message": (
                "Regression models and classifiers disagree on direction. "
                "Exercise caution — mixed signals detected."
            ),
            "timestamp": now,
        })

    # 3. Regime transition alert
    safe_name = ticker.replace(".", "_")
    regime_history_path = DATA_DIR / safe_name / "regime_history.csv"
    if regime_history_path.exists():
        import pandas as pd
        rh = pd.read_csv(regime_history_path)
        if len(rh) >= 2:
            prev_regime = rh["regime"].iloc[-2]
            if prev_regime != current_regime:
                alerts.append({
                    "type": "REGIME_CHANGE",
                    "severity": "info",
                    "ticker": ticker,
                    "company": company,
                    "title": f"🔄 Regime shift for {company}",
                    "message": f"Market regime changed from {prev_regime} to {current_regime}.",
                    "timestamp": now,
                })

    # 4. High volatility alert
    if current_regime == "HIGH_VOLATILITY":
        alerts.append({
            "type": "HIGH_VOLATILITY",
            "severity": "danger",
            "ticker": ticker,
            "company": company,
            "title": f"🌊 High volatility for {company}",
            "message": "Market is in HIGH VOLATILITY regime. Increased risk — consider reducing position sizes.",
            "timestamp": now,
        })

    # 5. Large predicted move
    if abs(change_pct) > 3.0:
        alerts.append({
            "type": "LARGE_MOVE",
            "severity": "warning",
            "ticker": ticker,
            "company": company,
            "title": f"🚀 Large predicted move for {company}",
            "message": f"Models predict a {'+' if change_pct > 0 else ''}{change_pct:.1f}% move. Verify with additional analysis.",
            "timestamp": now,
        })

    # Save alerts
    save_alerts(ticker, alerts)
    return alerts


def save_alerts(ticker: str, new_alerts: list):
    """Save alerts to JSON file."""
    alerts_path = DATA_DIR / "alerts.json"
    existing = []

    if alerts_path.exists():
        with open(alerts_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.extend(new_alerts)
    # Keep last 200 alerts
    existing = existing[-200:]

    with open(alerts_path, "w") as f:
        json.dump(existing, f, indent=2)


def get_recent_alerts(limit: int = 20) -> list:
    """Get recent alerts across all stocks."""
    alerts_path = DATA_DIR / "alerts.json"
    if not alerts_path.exists():
        return []

    with open(alerts_path, "r") as f:
        try:
            alerts = json.load(f)
        except json.JSONDecodeError:
            return []

    return alerts[-limit:]
