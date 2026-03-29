"""
Apex-Oracle — Logistic Regression Classifier
Baseline classifier with interpretable coefficients.
Outputs calibrated probabilities for direction.
"""
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import DATA_DIR, MODELS_DIR, STOCK_UNIVERSE


class LogisticRegressionModel:
    """Logistic Regression with L2 regularization for direction classification."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        exclude = ["Date", "direction", "Close"]
        return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]]

    def train(self, ticker: str) -> dict:
        safe_name = ticker.replace(".", "_")
        features_path = DATA_DIR / safe_name / "features.csv"
        if not features_path.exists():
            return {"status": "error", "message": "No features file"}

        df = pd.read_csv(features_path, parse_dates=["Date"])
        df = df.iloc[:-1]
        self.feature_cols = self._get_feature_cols(df)

        X = df[self.feature_cols].values
        y = df["direction"].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "feature_cols": self.feature_cols
        }, MODELS_DIR / f"logistic_regression_{safe_name}.pkl")

        logger.info(f"[{ticker}] Logistic Regression — Accuracy: {accuracy:.2%}")
        return {"status": "ok", "accuracy": accuracy}

    def predict(self, ticker: str) -> dict:
        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"logistic_regression_{safe_name}.pkl"
        if not model_path.exists():
            self.train(ticker)

        saved = joblib.load(model_path)
        self.model, self.scaler = saved["model"], saved["scaler"]
        self.feature_cols = saved["feature_cols"]

        df = pd.read_csv(DATA_DIR / safe_name / "features.csv")
        X_latest = self.scaler.transform(df[self.feature_cols].values[-1:])

        direction = int(self.model.predict(X_latest)[0])
        probabilities = self.model.predict_proba(X_latest)[0]

        return {
            "direction": "UP" if direction == 1 else "DOWN",
            "probability_up": float(probabilities[1]) if len(probabilities) > 1 else float(direction),
            "probability_down": float(probabilities[0]) if len(probabilities) > 1 else float(1 - direction),
            "model": "Logistic Regression",
        }

    def get_coefficients(self, ticker: str) -> dict:
        """Get model coefficients for interpretability."""
        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"logistic_regression_{safe_name}.pkl"
        if not model_path.exists():
            return {}

        saved = joblib.load(model_path)
        model = saved["model"]
        feature_cols = saved["feature_cols"]
        coefficients = model.coef_[0]
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        return {feature_cols[i]: float(coefficients[i]) for i in sorted_idx[:15]}
