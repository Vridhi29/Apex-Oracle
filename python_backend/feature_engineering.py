"""
Apex-Oracle — Feature Engineering
Computes technical indicators, lag features, and merges sentiment scores.
Outputs a feature matrix for each stock.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from config import DATA_DIR, STOCK_UNIVERSE, TECHNICAL_INDICATORS


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators from OHLCV data."""
    import ta

    df = df.copy()

    # Ensure numeric columns
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])

    # ── RSI ─────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # ── MACD ────────────────────────────────────────
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # ── Bollinger Bands ────────────────────────────
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # ── ATR (Average True Range) ───────────────────
    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()

    # ── Moving Averages ────────────────────────────
    df["ema_12"] = ta.trend.EMAIndicator(df["Close"], window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(df["Close"], window=26).ema_indicator()
    df["sma_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()

    # ── Volume Ratio ───────────────────────────────
    df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"].replace(0, np.nan)

    # ── Momentum & Rate of Change ──────────────────
    df["momentum"] = df["Close"] - df["Close"].shift(10)
    df["roc"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100

    # ── Daily Returns ──────────────────────────────
    df["daily_return"] = df["Close"].pct_change()

    # ── Rolling Volatility (20-day) ────────────────
    df["volatility_20d"] = df["daily_return"].rolling(window=20).std()

    # ── Price Change Direction (Target for classifiers) ──
    df["direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    # 1 = price went UP next day, 0 = price went DOWN

    return df


def create_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """Create lag features for Close price and key indicators."""
    df = df.copy()
    if lags is None:
        lags = [1, 2, 3, 5, 7, 10]

    for lag in lags:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
        df[f"volume_lag_{lag}"] = df["Volume"].shift(lag)

    # Rolling statistics
    df["close_rolling_mean_5"] = df["Close"].rolling(5).mean()
    df["close_rolling_std_5"] = df["Close"].rolling(5).std()
    df["close_rolling_mean_10"] = df["Close"].rolling(10).mean()
    df["close_rolling_std_10"] = df["Close"].rolling(10).std()

    return df


def merge_sentiment_scores(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Merge daily sentiment scores into the feature matrix."""
    safe_name = ticker.replace(".", "_")
    sentiment_path = DATA_DIR / safe_name / "sentiment_scores.csv"

    if sentiment_path.exists():
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=["Date"])
        df = df.merge(sentiment_df, on="Date", how="left")
        # Forward-fill sentiment for days without news
        sentiment_cols = [c for c in sentiment_df.columns if c != "Date"]
        df[sentiment_cols] = df[sentiment_cols].fillna(method="ffill").fillna(0)
        logger.info(f"[{ticker}] Merged sentiment scores")
    else:
        # Default sentiment columns
        df["sentiment_score"] = 0.0
        df["sentiment_positive"] = 0.0
        df["sentiment_negative"] = 0.0
        df["sentiment_neutral"] = 1.0
        logger.info(f"[{ticker}] No sentiment data found, using defaults")

    return df

def merge_fundamental_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Merge fundamental data into the feature matrix."""
    import json
    safe_name = ticker.replace(".", "_")
    fund_path = DATA_DIR / safe_name / "fundamentals.json"
    
    # Defaults
    df["pe_ratio"] = 0.0
    df["earnings_growth"] = 0.0
    df["revenue_growth"] = 0.0
    df["market_cap"] = 0.0

    if fund_path.exists():
        try:
            with open(fund_path, "r") as f:
                funds = json.load(f)
                df["pe_ratio"] = float(funds.get("pe_ratio") or 0.0)
                df["earnings_growth"] = float(funds.get("earnings_growth") or 0.0)
                df["revenue_growth"] = float(funds.get("quarterly_revenue_growth") or 0.0)
                df["market_cap"] = float(funds.get("market_cap") or 0.0)
            logger.info(f"[{ticker}] Merged fundamental data")
        except Exception as e:
            logger.error(f"[{ticker}] Failed to merge fundamentals: {e}")
    else:
        logger.info(f"[{ticker}] No fundamental data found, using defaults")

    return df


def build_features(ticker: str) -> pd.DataFrame:
    """Full feature engineering pipeline for a single stock."""
    safe_name = ticker.replace(".", "_")
    raw_path = DATA_DIR / safe_name / "raw_data.csv"

    if not raw_path.exists():
        logger.warning(f"[{ticker}] No raw data found at {raw_path}")
        return pd.DataFrame()

    # Load raw data
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    logger.info(f"[{ticker}] Loaded {len(df)} rows")

    # Compute technical indicators
    df = compute_technical_indicators(df)

    # Create lag features
    df = create_lag_features(df)

    # Merge sentiment
    df = merge_sentiment_scores(df, ticker)

    # Merge fundamentals
    df = merge_fundamental_data(df, ticker)

    # Drop rows with NaN from rolling calculations
    df = df.dropna().reset_index(drop=True)

    # Save features
    features_path = DATA_DIR / safe_name / "features.csv"
    df.to_csv(features_path, index=False)
    logger.info(f"[{ticker}] Saved {len(df)} feature rows to {features_path}")

    return df


def build_all_features():
    """Build features for all stocks in the universe."""
    results = {}
    for ticker in STOCK_UNIVERSE:
        df = build_features(ticker)
        results[ticker] = len(df) if not df.empty else 0
        logger.info(f"[{ticker}] Features: {results[ticker]} rows")

    return results


if __name__ == "__main__":
    build_all_features()
