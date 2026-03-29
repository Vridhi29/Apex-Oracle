"""
Apex-Oracle — Random Forest Classifier
Ensemble of decision trees for direction classification.
Robust against overfitting, provides feature importance.
"""
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from config import DATA_DIR, MODELS_DIR, STOCK_UNIVERSE


class RandomForestModel:
    """Random Forest for direction classification."""

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

        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=5,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        logger.info(f"[{ticker}] RF Best Params: {grid_search.best_params_}")

        y_pred = self.model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "feature_cols": self.feature_cols
        }, MODELS_DIR / f"random_forest_{safe_name}.pkl")

        logger.info(f"[{ticker}] Random Forest — Accuracy: {accuracy:.2%}")
        return {"status": "ok", "accuracy": accuracy}

    def predict(self, ticker: str) -> dict:
        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"random_forest_{safe_name}.pkl"
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
            "model": "Random Forest",
        }

    def get_feature_importance(self, ticker: str) -> dict:
        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"random_forest_{safe_name}.pkl"
        if not model_path.exists():
            return {}

        saved = joblib.load(model_path)
        model = saved["model"]
        feature_cols = saved["feature_cols"]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        return {feature_cols[i]: float(importances[i]) for i in sorted_idx[:15]}
