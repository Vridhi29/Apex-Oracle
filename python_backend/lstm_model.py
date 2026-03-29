"""
Apex-Oracle — LSTM Model (Regression)
Deep learning model for price prediction using sequential features.
Best in trending (BULL/BEAR) regimes.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

from config import DATA_DIR, MODELS_DIR, STOCK_UNIVERSE, LOOKBACK_DAYS


class LSTMModel:
    """LSTM neural network for price prediction."""

    def __init__(self, lookback: int = LOOKBACK_DAYS):
        self.lookback = lookback
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Select features for LSTM input."""
        exclude = ["Date", "direction", "Close"]
        return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]]

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sliding window sequences for LSTM."""
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i - self.lookback:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def train(self, ticker: str) -> dict:
        """Train LSTM on a stock's feature data."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow-cpu")
            return {"status": "error", "message": "TensorFlow not available"}

        safe_name = ticker.replace(".", "_")
        features_path = DATA_DIR / safe_name / "features.csv"

        if not features_path.exists():
            return {"status": "error", "message": "No features file"}

        df = pd.read_csv(features_path, parse_dates=["Date"])
        self.feature_cols = self._get_feature_cols(df)

        X_data = df[self.feature_cols].values
        y_data = df["Close"].values.reshape(-1, 1)

        # Scale
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        X_scaled = self.scaler_X.fit_transform(X_data)
        y_scaled = self.scaler_y.fit_transform(y_data)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled.flatten())

        if len(X_seq) < 100:
            return {"status": "error", "message": "Not enough data for LSTM"}

        # Train/test split (80/20)
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        # Build model
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.lookback, len(self.feature_cols))),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")

        # Train
        self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )

        # Evaluate
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        y_test_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mae = float(np.mean(np.abs(y_test_actual - y_pred_actual)))
        mape = float(np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(str(MODELS_DIR / f"lstm_{safe_name}.keras"))
        joblib.dump({
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "feature_cols": self.feature_cols,
        }, MODELS_DIR / f"lstm_{safe_name}_scalers.pkl")

        logger.info(f"[{ticker}] LSTM trained — MAE: ₹{mae:.2f}, MAPE: {mape:.2f}%")
        return {"status": "ok", "mae": mae, "mape": mape}

    def predict(self, ticker: str) -> dict:
        """Predict next-day closing price."""
        try:
            import tensorflow as tf
        except ImportError:
            return {"prediction": None, "error": "TensorFlow not available"}

        safe_name = ticker.replace(".", "_")
        model_path = MODELS_DIR / f"lstm_{safe_name}.keras"
        scaler_path = MODELS_DIR / f"lstm_{safe_name}_scalers.pkl"

        if not model_path.exists():
            logger.warning(f"[{ticker}] No LSTM model, training...")
            self.train(ticker)

        self.model = tf.keras.models.load_model(str(model_path))
        saved = joblib.load(scaler_path)
        self.scaler_X = saved["scaler_X"]
        self.scaler_y = saved["scaler_y"]
        self.feature_cols = saved["feature_cols"]

        # Load latest features
        features_path = DATA_DIR / safe_name / "features.csv"
        df = pd.read_csv(features_path)
        X_latest = df[self.feature_cols].values[-self.lookback:]
        X_scaled = self.scaler_X.transform(X_latest)
        X_input = X_scaled.reshape(1, self.lookback, len(self.feature_cols))

        # Predict
        pred_scaled = self.model.predict(X_input, verbose=0)[0][0]
        prediction = float(self.scaler_y.inverse_transform([[pred_scaled]])[0][0])
        current_price = float(df["Close"].iloc[-1])
        change_pct = (prediction - current_price) / current_price * 100

        return {
            "prediction": prediction,
            "current_price": current_price,
            "change_pct": change_pct,
            "direction": "UP" if change_pct > 0 else "DOWN",
            "model": "LSTM",
        }


if __name__ == "__main__":
    model = LSTMModel()
    for ticker in STOCK_UNIVERSE:
        result = model.train(ticker)
        logger.info(f"{ticker}: {result}")
