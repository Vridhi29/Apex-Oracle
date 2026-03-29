"""
Apex-Oracle — Market Regime Detection Engine
Uses Hidden Markov Model to classify market into regimes:
BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

from config import DATA_DIR, MODELS_DIR, STOCK_UNIVERSE

REGIME_LABELS = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS", 3: "HIGH_VOLATILITY"}


class RegimeDetector:
    """HMM-based market regime classifier."""

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = None

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract regime-relevant features from price data."""
        features = pd.DataFrame()

        # 20-day rolling volatility of returns
        features["volatility"] = df["daily_return"].rolling(20).std()

        # Rate of change (momentum)
        features["momentum"] = df["roc"] if "roc" in df.columns else (
            (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100
        )

        # Volume ratio
        features["volume_ratio"] = df["volume_ratio"] if "volume_ratio" in df.columns else (
            df["Volume"] / df["Volume"].rolling(20).mean()
        )

        # Trend strength (EMA crossover)
        if "ema_12" in df.columns and "ema_26" in df.columns:
            features["trend_strength"] = (df["ema_12"] - df["ema_26"]) / df["ema_26"] * 100
        else:
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            features["trend_strength"] = (ema12 - ema26) / ema26 * 100

        # RSI deviation from 50 (neutral)
        if "rsi" in df.columns:
            features["rsi_deviation"] = df["rsi"] - 50
        else:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            features["rsi_deviation"] = (100 - 100 / (1 + rs)) - 50

        features = features.dropna()
        return features

    def train(self, ticker: str) -> dict:
        """Train the HMM on a stock's historical data."""
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        safe_name = ticker.replace(".", "_")
        features_path = DATA_DIR / safe_name / "features.csv"

        if not features_path.exists():
            logger.warning(f"[{ticker}] No features file for regime training")
            return {"status": "error", "message": "No features file"}

        df = pd.read_csv(features_path, parse_dates=["Date"])
        features = self._prepare_features(df)

        if len(features) < 100:
            logger.warning(f"[{ticker}] Not enough data for regime training ({len(features)} rows)")
            return {"status": "error", "message": "Not enough data"}

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)

        # Train HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.model.fit(X)

        # Predict regimes for all historical data
        regimes = self.model.predict(X)

        # Map HMM states to regime labels based on characteristics
        regime_mapping = self._map_regimes_to_labels(features, regimes)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"hmm_{safe_name}.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "regime_mapping": regime_mapping
        }, model_path)

        logger.info(f"[{ticker}] Regime detector trained, saved to {model_path}")

        # Save regime history
        regime_df = pd.DataFrame({
            "Date": df["Date"].iloc[-len(regimes):].values,
            "regime_id": regimes,
            "regime": [REGIME_LABELS[regime_mapping[r]] for r in regimes]
        })
        regime_path = DATA_DIR / safe_name / "regime_history.csv"
        regime_df.to_csv(regime_path, index=False)

        return {
            "status": "ok",
            "regime_distribution": {
                REGIME_LABELS[regime_mapping[i]]: int(np.sum(regimes == i))
                for i in range(self.n_regimes)
            }
        }

    def _map_regimes_to_labels(self, features: pd.DataFrame, regimes: np.ndarray) -> dict:
        """
        Map HMM state numbers to meaningful regime labels.
        Uses the characteristics of each state to determine the label.
        """
        state_stats = {}
        for state in range(self.n_regimes):
            mask = regimes == state
            if np.sum(mask) == 0:
                state_stats[state] = {"volatility": 0, "momentum": 0}
                continue
            state_stats[state] = {
                "volatility": features.iloc[mask]["volatility"].mean() if "volatility" in features.columns else 0,
                "momentum": features.iloc[mask]["momentum"].mean() if "momentum" in features.columns else 0,
            }

        # Sort states by volatility and momentum
        states = list(range(self.n_regimes))

        # Highest volatility → HIGH_VOLATILITY (3)
        high_vol = max(states, key=lambda s: state_stats[s]["volatility"])

        # Remaining: highest positive momentum → BULL (0)
        remaining = [s for s in states if s != high_vol]
        bull = max(remaining, key=lambda s: state_stats[s]["momentum"])

        # Remaining: most negative momentum → BEAR (1)
        remaining = [s for s in remaining if s != bull]
        bear = min(remaining, key=lambda s: state_stats[s]["momentum"])

        # Last one → SIDEWAYS (2)
        remaining = [s for s in remaining if s != bear]
        sideways = remaining[0] if remaining else bear

        mapping = {
            bull: 0,        # BULL
            bear: 1,        # BEAR
            sideways: 2,    # SIDEWAYS
            high_vol: 3,    # HIGH_VOLATILITY
        }

        return mapping

    def predict_current_regime(self, ticker: str) -> dict:
        """Predict the current market regime for a stock."""
        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"hmm_{safe_name}.pkl"

        if not model_path.exists():
            logger.warning(f"[{ticker}] No trained regime model, training now...")
            self.train(ticker)
            if not model_path.exists():
                return {"regime": "SIDEWAYS", "confidence": 0.0}

        # Load model
        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.scaler = saved["scaler"]
        regime_mapping = saved["regime_mapping"]

        # Load features
        features_path = DATA_DIR / safe_name / "features.csv"
        if not features_path.exists():
            return {"regime": "SIDEWAYS", "confidence": 0.0}

        df = pd.read_csv(features_path, parse_dates=["Date"])
        features = self._prepare_features(df)

        if len(features) < 1:
            return {"regime": "SIDEWAYS", "confidence": 0.0}

        # Use last data point for current regime
        X = self.scaler.transform(features.values[-1:])
        regime_id = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        mapped_regime = REGIME_LABELS[regime_mapping[regime_id]]
        confidence = float(probabilities[regime_id])

        return {
            "regime": mapped_regime,
            "confidence": confidence,
            "probabilities": {
                REGIME_LABELS[regime_mapping[i]]: float(probabilities[i])
                for i in range(self.n_regimes)
            }
        }


def train_all_regimes():
    """Train regime detectors for all stocks."""
    detector = RegimeDetector()
    results = {}
    for ticker in STOCK_UNIVERSE:
        result = detector.train(ticker)
        results[ticker] = result
        logger.info(f"[{ticker}] Regime training: {result}")
    return results


if __name__ == "__main__":
    train_all_regimes()
