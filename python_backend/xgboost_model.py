"""
Apex-Oracle — XGBoost Model (Regression)
Gradient-boosted trees for price prediction.
Strong at capturing non-linear feature interactions.
"""
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from config import DATA_DIR, MODELS_DIR, STOCK_UNIVERSE


class XGBoostModel:
    """XGBoost regressor for price prediction."""

    def __init__(self):
        self.model = None
        self.feature_cols = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        exclude = ["Date", "direction", "Close"]
        return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]]

    def train(self, ticker: str) -> dict:
        """Train XGBoost on a stock's feature data."""
        try:
            import xgboost as xgb
        except ImportError:
            return {"status": "error", "message": "xgboost not installed"}

        safe_name = ticker.replace(".", "_")
        features_path = DATA_DIR / safe_name / "features.csv"

        if not features_path.exists():
            return {"status": "error", "message": "No features file"}

        # We want to predict TOMORROW'S close using today's features
        df["target"] = df["Close"].shift(-1)
        df = df.dropna(subset=["target"])
        
        self.feature_cols = self._get_feature_cols(df)

        X = df[self.feature_cols].values
        y = df["target"].values

        # Train/test split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        grid_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=5, # fast random search
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        logger.info(f"[{ticker}] Best XGBoost Params: {grid_search.best_params_}")

        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = float(np.mean(np.abs(y_test - y_pred)))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
        }, MODELS_DIR / f"xgboost_{safe_name}.pkl")

        logger.info(f"[{ticker}] XGBoost trained — MAE: ₹{mae:.2f}, MAPE: {mape:.2f}%")
        return {"status": "ok", "mae": mae, "mape": mape}

    def predict(self, ticker: str) -> dict:
        """Predict next-day closing price."""
        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"xgboost_{safe_name}.pkl"

        if not model_path.exists():
            self.train(ticker)

        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.feature_cols = saved["feature_cols"]

        features_path = DATA_DIR / safe_name / "features.csv"
        df = pd.read_csv(features_path)
        X_latest = df[self.feature_cols].values[-1:]

        prediction = float(self.model.predict(X_latest)[0])
        current_price = float(df["Close"].iloc[-1])
        change_pct = (prediction - current_price) / current_price * 100

        return {
            "prediction": prediction,
            "current_price": current_price,
            "change_pct": change_pct,
            "direction": "UP" if change_pct > 0 else "DOWN",
            "model": "XGBoost",
        }

    def get_feature_importance(self, ticker: str) -> dict:
        """Get feature importance for explainability."""
        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"xgboost_{safe_name}.pkl"

        if not model_path.exists():
            return {}

        saved = joblib.load(model_path)
        model = saved["model"]
        feature_cols = saved["feature_cols"]

        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        return {
            feature_cols[i]: float(importances[i])
            for i in sorted_idx[:15]
        }


if __name__ == "__main__":
    model = XGBoostModel()
    for ticker in STOCK_UNIVERSE:
        result = model.train(ticker)
        logger.info(f"{ticker}: {result}")
