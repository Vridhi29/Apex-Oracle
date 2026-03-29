"""
Apex-Oracle — Dual-Layer Fusion Engine
THE NOVEL CORE: Combines regression price predictions + classification direction votes
with regime-aware dynamic weighting.
"""
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime
import gc
import sqlite3
import shutil
import glob

from config import (
    DATA_DIR, MODELS_DIR, R_MODEL_DIR, STOCK_UNIVERSE,
    REGIME_WEIGHTS, HIGH_CONFIDENCE_THRESHOLD
)
from regime_detector import RegimeDetector
from lstm_model import LSTMModel
from xgboost_model import XGBoostModel
from naive_bayes_model import NaiveBayesModel
from svm_model import SVMModel
from random_forest_model import RandomForestModel
from logistic_regression_model import LogisticRegressionModel


class FusionEngine:
    """
    Dual-Layer Confidence-Weighted Dynamic Fusion Engine.
    Layer 1: Regression models predict price
    Layer 2: Classification models vote on direction
    Both layers combined with regime-aware weighting.
    """

    def __init__(self):
        self.regime_detector = RegimeDetector()
        # Regression models
        self.lstm = LSTMModel()
        self.xgboost = XGBoostModel()
        # Classification models
        self.naive_bayes = NaiveBayesModel()
        self.svm = SVMModel()
        self.random_forest = RandomForestModel()
        self.logistic_regression = LogisticRegressionModel()

    def _run_r_model(self, script_name: str, ticker: str) -> dict:
        """Execute an R model script and parse JSON output."""
        safe_name = ticker.replace(".", "_")
        csv_path = str(DATA_DIR / safe_name / "features.csv")
        script_path = str(R_MODEL_DIR / script_name)

        try:
            rscript_exe = "Rscript"
            if not shutil.which("Rscript"):
                r_paths = glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
                if r_paths:
                    rscript_exe = sorted(r_paths)[-1]
                    
            result = subprocess.run(
                [rscript_exe, script_path, csv_path, "5"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            else:
                logger.warning(f"R script {script_name} error: {result.stderr[:200]}")
                return {"status": "error", "error": result.stderr[:200]}
        except FileNotFoundError:
            logger.warning("Rscript not found — R models will be skipped")
            return {"status": "error", "error": "Rscript not found"}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "R script timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_all_predictions(self, ticker: str) -> dict:
        """Get predictions from all 8 models."""
        results = {
            "regression": {},
            "classification": {},
        }

        # ── Layer 1: Regression Models ──────────────────

        # ARIMA (R)
        logger.info(f"[{ticker}] Running ARIMA...")
        arima_result = self._run_r_model("arima_model.R", ticker)
        if arima_result.get("status") == "ok":
            results["regression"]["arima"] = {
                "prediction": arima_result["next_day_prediction"],
                "change_pct": arima_result["change_pct"],
                "direction": arima_result["direction"],
            }
        gc.collect()

        # GARCH (R)
        logger.info(f"[{ticker}] Running GARCH...")
        garch_result = self._run_r_model("garch_model.R", ticker)
        if garch_result.get("status") == "ok":
            results["regression"]["garch"] = {
                "prediction": garch_result["next_day_prediction"],
                "change_pct": garch_result["change_pct"],
                "direction": garch_result["direction"],
                "volatility": garch_result.get("next_day_volatility", 0),
            }
        gc.collect()

        # LSTM (Python)
        logger.info(f"[{ticker}] Running LSTM...")
        try:
            lstm_result = self.lstm.predict(ticker)
            if lstm_result.get("prediction"):
                results["regression"]["lstm"] = lstm_result
        except Exception as e:
            logger.warning(f"LSTM error: {e}")
        gc.collect()

        # XGBoost (Python)
        logger.info(f"[{ticker}] Running XGBoost...")
        try:
            xgb_result = self.xgboost.predict(ticker)
            if xgb_result.get("prediction"):
                results["regression"]["xgboost"] = xgb_result
        except Exception as e:
            logger.warning(f"XGBoost error: {e}")
        gc.collect()

        # ── Layer 2: Classification Models ──────────────

        # Naive Bayes
        logger.info(f"[{ticker}] Running Naive Bayes...")
        try:
            results["classification"]["naive_bayes"] = self.naive_bayes.predict(ticker)
        except Exception as e:
            logger.warning(f"Naive Bayes error: {e}")

        # SVM
        logger.info(f"[{ticker}] Running SVM...")
        try:
            results["classification"]["svm"] = self.svm.predict(ticker)
        except Exception as e:
            logger.warning(f"SVM error: {e}")

        # Random Forest
        logger.info(f"[{ticker}] Running Random Forest...")
        try:
            results["classification"]["random_forest"] = self.random_forest.predict(ticker)
        except Exception as e:
            logger.warning(f"Random Forest error: {e}")

        # Logistic Regression
        logger.info(f"[{ticker}] Running Logistic Regression...")
        try:
            results["classification"]["logistic_regression"] = self.logistic_regression.predict(ticker)
        except Exception as e:
            logger.warning(f"Logistic Regression error: {e}")

        gc.collect()
        return results

    def fuse_predictions(self, ticker: str) -> dict:
        """
        THE CORE ALGORITHM: Dual-layer fusion with regime-aware weighting.
        """
        # Step 1: Detect current regime
        regime_info = self.regime_detector.predict_current_regime(ticker)
        regime = regime_info.get("regime", "SIDEWAYS")
        regime_confidence = regime_info.get("confidence", 0.5)

        # Step 2: Get all model predictions
        predictions = self.get_all_predictions(ticker)

        # Step 3: Layer 1 — Weighted regression ensemble
        regression_weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["SIDEWAYS"])
        weighted_prices = []
        total_weight = 0
        regression_directions = []

        for model_name, weight in regression_weights.items():
            if model_name in predictions["regression"]:
                pred = predictions["regression"][model_name]
                weighted_prices.append(pred["prediction"] * weight)
                total_weight += weight
                regression_directions.append(pred["direction"])

        if total_weight > 0:
            ensemble_price = sum(weighted_prices) / total_weight
        else:
            # Fallback: use any available prediction
            any_pred = list(predictions["regression"].values())
            ensemble_price = any_pred[0]["prediction"] if any_pred else 0

        # Step 4: Layer 2 — Classification probability-weighted voting
        up_weight = 0.0
        down_weight = 0.0
        classifier_names = []

        for model_name, pred in predictions["classification"].items():
            prob_up = pred.get("probability_up", 0.5 if pred["direction"] == "UP" else 0.49)
            up_weight += prob_up
            down_weight += (1.0 - prob_up)
            classifier_names.append(model_name)

        total_classifiers = len(classifier_names)
        classification_direction = "UP" if up_weight > down_weight else "DOWN"
        classification_confidence = max(up_weight, down_weight) / max(total_classifiers, 1)

        avg_prob_up = up_weight / max(total_classifiers, 1) if total_classifiers > 0 else 0.5

        # Step 5: Cross-layer agreement check
        # Get regression direction
        current_price = 0
        for pred in predictions["regression"].values():
            current_price = pred.get("current_price", 0)
            break

        regression_direction = "UP" if ensemble_price > current_price else "DOWN"
        cross_layer_agrees = (regression_direction == classification_direction)

        # Step 6: Compute final confidence
        # Base confidence from regression model agreement
        if regression_directions:
            reg_up = sum(1 for d in regression_directions if d == "UP")
            reg_agreement = max(reg_up, len(regression_directions) - reg_up) / len(regression_directions)
        else:
            reg_agreement = 0.5

        # Combined confidence
        confidence = (
            0.35 * reg_agreement +
            0.30 * classification_confidence +
            0.20 * regime_confidence +
            0.15 * (1.0 if cross_layer_agrees else 0.3)
        )

        # Boost/penalize based on cross-layer agreement
        if cross_layer_agrees:
            confidence = min(confidence * 1.15, 0.95)
        else:
            confidence *= 0.75

        conflict_flag = not cross_layer_agrees

        # Total models agreeing on direction
        all_directions = regression_directions + [
            p["direction"] for p in predictions["classification"].values()
        ]
        final_direction = regression_direction  # Trust price prediction
        models_agreeing = sum(1 for d in all_directions if d == final_direction)

        change_pct = (ensemble_price - current_price) / current_price * 100 if current_price > 0 else 0

        # Step 7: Package result
        fusion_result = {
            "ticker": ticker,
            "company": STOCK_UNIVERSE.get(ticker, ticker),
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "price": round(ensemble_price, 2),
                "current_price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "direction": final_direction,
                "confidence": round(confidence, 4),
                "confidence_label": (
                    "High" if confidence >= 0.8 else
                    "Medium" if confidence >= 0.6 else "Low"
                ),
            },
            "regime": {
                "current": regime,
                "confidence": round(regime_confidence, 4),
                "probabilities": regime_info.get("probabilities", {}),
            },
            "models": {
                "regression": predictions["regression"],
                "classification": predictions["classification"],
                "regression_weights": regression_weights,
            },
            "agreement": {
                "total_models": len(all_directions),
                "models_agreeing": models_agreeing,
                "cross_layer_agrees": cross_layer_agrees,
                "conflict_flag": conflict_flag,
                "classification_vote": f"{up_weight:.2f} UP / {down_weight:.2f} DOWN",
            },
        }

        # Save result
        safe_name = ticker.replace(".", "_")
        result_path = DATA_DIR / safe_name / "latest_prediction.json"
        with open(result_path, "w") as f:
            json.dump(fusion_result, f, indent=2)

        # Append to SQLite for historical backtesting
        try:
            history_db = DATA_DIR / "predictions_history.db"
            conn = sqlite3.connect(history_db)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            ticker TEXT,
                            predicted_price REAL,
                            actual_price REAL,
                            predicted_direction TEXT,
                            confidence REAL
                        )''')
            c.execute("INSERT INTO predictions (timestamp, ticker, predicted_price, actual_price, predicted_direction, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                      (fusion_result["timestamp"], ticker, ensemble_price, current_price, final_direction, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[{ticker}] Failed to save prediction history: {e}")

        logger.info(
            f"[{ticker}] Prediction: ₹{ensemble_price:.2f} ({final_direction}, "
            f"conf: {confidence:.0%}) | Regime: {regime} | "
            f"Agreement: {models_agreeing}/{len(all_directions)}"
        )

        return fusion_result

    def fuse_all_stocks(self) -> dict:
        """Run fusion for all stocks in the universe."""
        results = {}
        for ticker in STOCK_UNIVERSE:
            try:
                results[ticker] = self.fuse_predictions(ticker)
            except Exception as e:
                logger.error(f"[{ticker}] Fusion error: {e}")
                results[ticker] = {"error": str(e)}
            gc.collect()
        return results


if __name__ == "__main__":
    engine = FusionEngine()
    for ticker in STOCK_UNIVERSE:
        result = engine.fuse_predictions(ticker)
        print(json.dumps(result, indent=2))
