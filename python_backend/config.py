"""
Apex-Oracle Configuration
Central configuration for all stocks, paths, and scheduler settings.
"""
import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
R_MODEL_DIR = PROJECT_ROOT / "r_model"

# ── Multi-Stock Universe ───────────────────────────────────
STOCK_UNIVERSE = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
}

DEFAULT_TICKER = "RELIANCE.NS"

# ── Data Fetching ──────────────────────────────────────────
HISTORICAL_START_DATE = "2015-01-01"
LOOKBACK_DAYS = 60  # Sliding window for LSTM

# ── Market Hours (IST) ────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# ── Scheduler Settings ─────────────────────────────────────
FETCH_INTERVAL_MINUTES = 5          # During market hours
POST_MARKET_INTERVAL_MINUTES = 30   # After market close
RETRAIN_DAY = "sun"                 # Weekly retrain day
RETRAIN_HOUR = 2                    # 2 AM Sunday

# ── Model Regime Weights ───────────────────────────────────
# Regression model weights per regime
REGIME_WEIGHTS = {
    "BULL": {
        "arima": 0.25, "garch": 0.10, "lstm": 0.35, "xgboost": 0.30
    },
    "BEAR": {
        "arima": 0.15, "garch": 0.25, "lstm": 0.25, "xgboost": 0.35
    },
    "SIDEWAYS": {
        "arima": 0.40, "garch": 0.15, "lstm": 0.15, "xgboost": 0.30
    },
    "HIGH_VOLATILITY": {
        "arima": 0.15, "garch": 0.40, "lstm": 0.20, "xgboost": 0.25
    },
}

# ── Confidence Thresholds ──────────────────────────────────
HIGH_CONFIDENCE_THRESHOLD = 0.80
ALERT_MIN_MODELS_AGREE = 6  # Out of 8 total models

# ── Feature Engineering ────────────────────────────────────
TECHNICAL_INDICATORS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "atr", "ema_12", "ema_26", "sma_20", "sma_50",
    "volume_ratio", "momentum", "roc"
]
